import numpy as np
import cv2
import torch
import math
import PIL.Image
from insightface.app import FaceAnalysis
import logging
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pipeline_ecomid import EcomIDPipeline
from pulid import attention_processor as attention
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import uvicorn
import io
from datetime import datetime
import uuid
import json
from pydantic import BaseModel
from typing import Optional
import facer
from transformers import pipeline
import mediapipe as mp




from huggingface_hub import hf_hub_download


device = 'cuda'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

insightface_app = None
pipeline_swap = None
face_parser = None
bg_remove_pipe = None
segmenter = None
executor = ThreadPoolExecutor(max_workers=1)

class HeadSegmentation:
    def __init__(self, model_path='selfie_multiclass_256x256.tflite'):
        """Initialize MediaPipe ImageSegmenter with multi-class selfie model."""
        try:
            if not os.path.exists(model_path):
                print(f"Downloading model to {model_path}...")
                model_path = self.download_model(model_path)
            self.options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_category_mask=True
            )
            self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(self.options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize segmenter: {str(e)}")
    def download_model(self, model_path):
        """Download the model from Hugging Face and verify its integrity."""
        repo_id = "yolain/selfie_multiclass_256x256"
        filename = "selfie_multiclass_256x256.tflite"

        # Download the file
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir='./')
        return downloaded_path
    def extract_head(self, image_input):
        """Extract head (hair + face-skin) from a PIL Image, returning RGBA PIL Image."""
        if not isinstance(image_input, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image object")
        
        try:
            # Handle alpha channel in input
            if image_input.mode == 'RGBA':
                print("Warning: Input image has alpha channel, converting to RGB")
            
            # Keep everything in RGB for consistency
            image_rgb = np.array(image_input.convert('RGB'))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Run segmentation
            segmentation_result = self.segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask
            if category_mask is None:
                raise ValueError("Segmentation failed: No category mask returned")
            
            # Extract category mask
            mask = category_mask.numpy_view()
            
            # Categories based on selfie_multiclass_256x256: 1=hair, 3=face-skin
            head_categories = [1, 3]
            binary_mask = np.zeros_like(mask, dtype=np.uint8)
            for cat in head_categories:
                binary_mask[mask == cat] = 255
            
            # Resize mask if needed
            if binary_mask.shape != image_rgb.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (image_rgb.shape[1], image_rgb.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                        
            extracted_image_rgb = image_rgb.copy()
            extracted_image_rgb[binary_mask == 0] = [0, 0, 0]  # Set non-head pixels to black
            
            # Create RGBA image
            extracted_image_rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
            extracted_image_rgba[:, :, :3] = extracted_image_rgb  # RGB channels
            extracted_image_rgba[:, :, 3] = binary_mask  # Alpha channel
            
            extracted_pil = PIL.Image.fromarray(extracted_image_rgba, 'RGBA')


            # Calculate bounding box
            coords = np.where(binary_mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                # PIL crop expects (left, top, right, bottom)
                bbox = (x_min, y_min, x_max + 1, y_max + 1)
            else:
                bbox = (0, 0, 0, 0)

            mask_pil = PIL.Image.fromarray(binary_mask, 'L').convert('RGB')

            return extracted_pil,bbox,mask_pil

        except Exception as e:
            raise RuntimeError(f"Segmentation error: {str(e)}")

    def close(self):
        """Clean up segmenter resources."""
        self.segmenter.close()




def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipeline_swap, insightface_app, face_parser,bg_remove_pipe,segmenter
    
    try:
        insightface_app = FaceAnalysis(name='antelopev2', root='./',
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        file_path = hf_hub_download(
            repo_id="lllyasviel/fav_models",
            filename="fav/realisticStockPhoto_v20.safetensors",
            local_dir="checkpoints/realisticStockPhoto_v20"
        )
        args = {
            'face_adapter_path': 'checkpoints/ip-adapter.bin',
            'controlnet_path': 'checkpoints/ControlNetModel',
            'base_model_path': file_path,
            'device': 'cuda:0'
        }
        pipeline_swap = EcomIDPipeline(args, insightface_app)
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
        device = torch.device(f'cuda:{0}')
        face_parser = facer.face_parser('farl/lapa/448', device=device)
        bg_remove_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device='cuda')
        segmenter = HeadSegmentation()




    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil




def make_head_transparent(original_image, extracted_head):
    """Make only the head pixels transparent using the extracted head mask"""
    # Convert original to RGBA if not already
    if original_image.mode != 'RGBA':
        result_image = original_image.convert('RGBA')
    else:
        result_image = original_image.copy()
    
    # Get alpha channel from extracted head
    head_alpha = extracted_head.split()[-1]  # Get alpha channel
    
    # Resize alpha mask to match original image size if needed
    if head_alpha.size != original_image.size:
        head_alpha = head_alpha.resize(original_image.size, PIL.Image.LANCZOS)
    
    # Convert to numpy arrays
    result_array = np.array(result_image)
    mask_array = np.array(head_alpha)
    
    # Where mask is white (255), make original transparent
    result_array[:, :, 3] = np.where(mask_array > 128, 0, result_array[:, :, 3])
    
    # Convert back to PIL
    transparent_image = PIL.Image.fromarray(result_array, 'RGBA')
    
    return transparent_image

def pred_info(img):
    info = insightface_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    info = max(info,key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
    return info

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, initialize_pipelines)
    yield


app = FastAPI(title="SDXL Face Swap API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    seed: Optional[int] = None
    strength: float = 0.9
    ip_adapter_scale: float = 0.5  # Lower for InstantID
    guidance_scale: float = 3.5  # Zero for LCM
    num_inference_steps: int = 30

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory job storage
jobs = {}
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)



async def gen_img2img(job_id: str, face_image : PIL.Image.Image,pose_image: PIL.Image.Image,request: Img2ImgRequest):
    print("Đang cắt vùng tóc và mặt...")
    head_img, bbox ,mask_head= segmenter.extract_head(
        pose_image, 
    )
    # from bbox crop pose_image
    top_layer_image = make_head_transparent(pose_image, head_img)
    crop_pose_image = pose_image.crop(bbox)
    mask_img = mask_head.crop(bbox)    

    width, height = crop_pose_image.size
    pose_info = pred_info(crop_pose_image)
    control_image = draw_kps(crop_pose_image, pose_info['kps'])
    face_info = pred_info(face_image)
    face_embed = np.array(face_info['embedding'])[None, ...]
    id_embeddings = pipeline_swap.get_id_embedding(np.array(face_image))
    image = pipeline_swap.inference(request.prompt, (1, height, width), control_image, face_embed, crop_pose_image, mask_img,
                             request.negative_prompt, id_embeddings, request.ip_adapter_scale, request.guidance_scale, request.num_inference_steps, request.strength)[0]
    filename = f"{job_id}_base.png"
    # create new PIL Image has size = top_layer_image
    new_generated_image = PIL.Image.new("RGBA", top_layer_image.size)
    x,y,_,_ = bbox
    new_generated_image.paste(image, (x, y))
    result_image = PIL.Image.new("RGBA", top_layer_image.size)
    result_image = PIL.Image.alpha_composite(new_generated_image, top_layer_image)    

    filepath = os.path.join(results_dir, filename)
   
    result_image.save(filepath)
        
    metadata = {
        "job_id": job_id,
        "type": "head_swap",
        "seed": 0,
        "prompt": request.prompt,
        "parameters": request.dict(),
        "filename": filename,
        "device_used": 'cuda',
    }
        
    metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 1.0
    jobs[job_id]["result_url"] = f"/results/{filename}"
    jobs[job_id]["metadata"] = metadata
    jobs[job_id]["completed_at"] = datetime.now()
    
    logger.info(f"Img2img completed successfully on cuda")








@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    try:
        with open("img2img_interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface_alt():
    """Alternative route for web interface"""
    return await serve_web_interface()

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved()
        }
    
    pipeline_device = None
    if pipeline_swap is not None:
        try:
            # Try to get device from unet (most reliable)
            pipeline_device = str(pipeline_swap.unet.device)
        except:
            try:
                # Fallback to vae device
                pipeline_device = str(pipeline_swap.vae.device)
            except:
                pipeline_device = "unknown"
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "pipeline_device": pipeline_device,
        "pipelines_loaded": pipeline_swap is not None,
        "face_analysis_loaded": insightface_app is not None,
        "gpu_info": gpu_info
    }




@app.post("/img2img")
async def img2img(
    base_image: UploadFile = File(...),
    pose_image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality"),
    strength: float = Form(0.9),
    ip_adapter_scale: float = Form(0.5),  # Lower for InstantID
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(3.5),  # Zero for LCM
    seed: Optional[int] = Form(None),
    
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "head_swap"
    }
    
    try:
        # Load images
        base_img = PIL.Image.open(io.BytesIO(await base_image.read())).convert('RGB')
        pose_img = PIL.Image.open(io.BytesIO(await pose_image.read())).convert('RGB')

        request = Img2ImgRequest(
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            guidance_scale=guidance_scale,
           
        )
        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_img2img(job_id, base_img, pose_img, request)
        ))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result_url": job.get("result_url"),
        "seed": job.get("metadata", {}).get("seed"),
        "error_message": job.get("error_message"),
        "created_at": job["created_at"].isoformat(),
        "completed_at": job.get("completed_at").isoformat() if job.get("completed_at") else None
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Get result image"""
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath)


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    try:
        job_list = []
        for job_id, job_data in jobs.items():
            job_list.append({
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "created_at": job_data.get("created_at", datetime.now()).isoformat(),
                "completed_at": job_data.get("completed_at").isoformat() if job_data.get("completed_at") else None,
                "result_url": job_data.get("result_url"),
                "error_message": job_data.get("error_message")
            })
        
        job_list.sort(key=lambda x: x["created_at"], reverse=True)
        return job_list
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return []

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job = jobs[job_id]
    if "metadata" in job and "filename" in job["metadata"]:
        filename = job["metadata"]["filename"]
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete metadata file
        metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    
    # Set environment variables for better CUDA error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=8888)