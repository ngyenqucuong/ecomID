import numpy as np
import cv2
import torch
import math
import PIL.Image
from insightface.app import FaceAnalysis
import logging
import os
import random
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

class HairFaceSegmentation:
    def __init__(self):
        # Khởi tạo MediaPipe Selfie Segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Khởi tạo MediaPipe Face Mesh để detect face landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_face_mask(self, image):
        """Tạo mask cho khuôn mặt dựa trên face landmarks"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy tọa độ các điểm trên khuôn mặt
                h, w = image.shape[:2]
                points = []
                
                # Outline của khuôn mặt (face oval)
                face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                for idx in face_oval_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
                
                # Tạo mask từ các điểm
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
        
        return mask

    def get_hair_mask(self, image, person_mask):
        """Tạo mask cho tóc bằng cách trừ face mask khỏi person mask"""
        face_mask = self.get_face_mask(image)
        
        # Dilate face mask một chút để loại bỏ phần cổ
        kernel = np.ones((20, 20), np.uint8)
        face_mask_dilated = cv2.dilate(face_mask, kernel, iterations=1)
        
        # Tóc = toàn bộ người - mặt (đã dilate)
        hair_mask = person_mask.copy()
        hair_mask[face_mask_dilated > 0] = 0
        
        # Chỉ giữ phần trên của mask (tóc thường ở phía trên)
        h = hair_mask.shape[0]
        hair_mask[int(h*0.7):] = 0  # Bỏ phần dưới 70% chiều cao
        
        return hair_mask

    def get_combined_mask(self, image_input):
        """Lấy mask kết hợp của tóc và mặt"""
        # Xử lý input (có thể là path hoặc PIL.Image)
        if isinstance(image_input, str):
            # Nếu là path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError("Không thể đọc ảnh")
        else:
            # Nếu là PIL.Image
            image_rgb_pil = image_input.convert('RGB')
            image = cv2.cvtColor(np.array(image_rgb_pil), cv2.COLOR_RGB2BGR)
        
        # Chuyển sang RGB cho MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy mask toàn bộ người
        results = self.selfie_segmentation.process(image_rgb)
        person_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Lấy mask mặt và tóc
        face_mask = self.get_face_mask(image)
        hair_mask = self.get_hair_mask(image, person_mask)
        
        # Kết hợp mask tóc và mặt
        combined_mask = cv2.bitwise_or(face_mask, hair_mask)
        
        return image, combined_mask, face_mask, hair_mask

    def extract_hair_face_region(self, image_input, output_path=None, padding=0):
        """Cắt vùng tóc và mặt ra khỏi ảnh"""
        image, combined_mask, face_mask, hair_mask = self.get_combined_mask(image_input)
        
        # Tìm bounding box của mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Không tìm thấy vùng tóc/mặt")
        
        # Lấy bounding box lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Thêm padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Cắt ảnh và mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = combined_mask[y:y+h, x:x+w]
        
        # Tạo ảnh trong suốt với alpha channel
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = cropped_image
        result[:, :, 3] = cropped_mask  # Alpha channel
        
        if output_path:
            # Lưu ảnh với alpha channel
            result_pil = PIL.Image.fromarray(result, 'RGBA')
            result_pil.save(output_path)
        
        return result, (x, y, w, h)  # Trả về ảnh và vị trí để gắn lại sau

    def blend_generated_image(self, original_image_input, generated_pil_image, 
                            bbox, output_path=None, blend_edges=True):
        """Gắn ảnh đã generate lại vào ảnh gốc"""
        # Xử lý ảnh gốc
        if isinstance(original_image_input, str):
            original = cv2.imread(original_image_input)
        else:
            # PIL.Image
            original_rgb_pil = original_image_input.convert('RGB')
            original = cv2.cvtColor(np.array(original_rgb_pil), cv2.COLOR_RGB2BGR)
        
        # Chuyển PIL Image sang numpy array
        if generated_pil_image.mode == 'RGBA':
            generated_array = np.array(generated_pil_image)
            generated_rgb = generated_array[:, :, :3]
            alpha = generated_array[:, :, 3] / 255.0
        else:
            generated_array = np.array(generated_pil_image.convert('RGB'))
            generated_rgb = generated_array
            alpha = np.ones(generated_array.shape[:2])
        
        # Chuyển từ RGB sang BGR cho OpenCV
        generated_rgb = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
        
        x, y, w, h = bbox
        
        # Resize generated image nếu cần
        if generated_rgb.shape[:2] != (h, w):
            generated_rgb = cv2.resize(generated_rgb, (w, h))
            alpha = cv2.resize(alpha, (w, h))
        
        # Tạo soft edges nếu được yêu cầu
        if blend_edges:
            kernel = np.ones((5, 5), np.float32) / 25
            alpha = cv2.filter2D(alpha, -1, kernel)
        
        # Blend ảnh
        result = original.copy()
        for c in range(3):
            result[y:y+h, x:x+w, c] = (
                alpha * generated_rgb[:, :, c] + 
                (1 - alpha) * original[y:y+h, x:x+w, c]
            )
        
        # Lưu kết quả nếu có output_path
        if output_path:
            cv2.imwrite(output_path, result)
        
        # Trả về PIL.Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(result_rgb)






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
        segmenter = HairFaceSegmentation()




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


def pred_face_mask(img, face_info):
    points = []
    rects = []
    scores = []
    image_ids = []
    # for i in range(len(detected_faces)):

    points.append(face_info['kps'])
    rects.append(face_info['bbox'])
    scores.append(face_info['det_score'])
    image_ids.append(0)

    face_info = {}
    face_info['points'] = torch.tensor(points).to(device)
    face_info['rects'] = torch.tensor(rects).to(device)
    face_info['scores'] = torch.tensor(scores).to(device)
    face_info['image_ids'] = torch.tensor(image_ids).to(device)

    img = np.array(img)
    reordered_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device=device)

    with torch.inference_mode():
        faces = face_parser(reordered_img, face_info)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
    vis_img = vis_seg_probs.sum(0, keepdim=True)

    face_mask = vis_img != 0

    # bbox = face_info['rects'][0]

    face_mask = face_mask.unsqueeze(0)
    face_mask = face_mask.squeeze().int()*255
    face_mask = face_mask.cpu().numpy()

    return face_mask



def prepareMaskAndPoseAndControlImage(pose_image, face_info,width,height):
    kps = face_info['kps']

    mask = np.zeros([height, width, 3])
    face_mask = pred_face_mask(pose_image, face_info)
    mask[face_mask>0] = 255
    face_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    face_mask = cv2.dilate(face_mask, kernel, iterations=2)
    face_mask = PIL.Image.fromarray(face_mask.astype(np.uint8))
    control_image = draw_kps(pose_image, kps)

    # (mask, pose, control PIL images), (original positon face + padding: x, y, w, h)
    return face_mask, control_image



def pred_face_info(img):
    face_info = insightface_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # only use the maximum face

    return face_info

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
    hair_face_region, bbox = segmenter.extract_hair_face_region(
        pose_image, 
        "hair_face_extracted.png"
    )
    
    # Chuyển sang PIL Image để đưa vào diffusion
    hair_face_pil = PIL.Image.fromarray(hair_face_region, 'RGBA').convert('RGB')
    width, height = hair_face_pil.size
    pose_info = insightface_app.get(cv2.cvtColor(np.array(hair_face_pil), cv2.COLOR_RGB2BGR))
    pose_info = max(pose_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    mask_image, control_image = prepareMaskAndPoseAndControlImage(
        hair_face_pil,
        pose_info,
        width,
        height
    )
    face_info = pred_face_info(face_image)
    face_embed = np.array(face_info['embedding'])[None, ...]
    id_embeddings = pipeline_swap.get_id_embedding(np.array(face_image))
    image = pipeline_swap.inference(request.prompt, (1, height, width), control_image, face_embed, hair_face_pil, mask_image,
                             request.negative_prompt, id_embeddings, request.ip_adapter_scale, request.guidance_scale, request.num_inference_steps, request.strength)[0]
    filename = f"{job_id}_base.png"
    
    filepath = os.path.join(results_dir, filename)
    segmenter.blend_generated_image(
        pose_image,
        image,  # PIL.Image thay vì path
        bbox,
        filepath
    )
    # image.save(filepath)
        
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