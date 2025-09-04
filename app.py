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
import onnxruntime
from basicsr.utils import img2tensor, tensor2img,imwrite
from torchvision.transforms.functional import normalize
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from transformers import AutoModelForImageSegmentation

from torchvision import transforms

from basicsr.utils.registry import ARCH_REGISTRY

from facelib.utils.face_restoration_helper import FaceRestoreHelper



from huggingface_hub import hf_hub_download


device = 'cuda'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

insightface_app = None
pipeline_swap = None
face_parser = None
bg_remove_pipe = None
rmodel = None
executor = ThreadPoolExecutor(max_workers=1)

upsampler = None

codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()


def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipeline_swap, insightface_app, face_parser,bg_remove_pipe,rmodel,upsampler
    
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
        attention.ORTHO_v2 = True
        device = torch.device(f'cuda:{0}')
        face_parser = facer.face_parser('farl/lapa/448', device=device)
        bg_remove_pipe = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        bg_remove_pipe.to('cuda')

        sess_options = onnxruntime.SessionOptions()
        rmodel = onnxruntime.InferenceSession('lama_fp32.onnx', sess_options=sess_options)
        
        upsampler = set_realesrgan()

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

def get_image(image):
    if isinstance(image, PIL.Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def predict(imagex, size):
    # Fix: Handle RMBG-2.0 output format
    input_images = transform_image(imagex).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = bg_remove_pipe(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    result = pred_pil.resize(size)
    # RMBG-2.0 returns a PIL Image directly
    if isinstance(result, PIL.Image.Image):
        mask = result.convert('L')
    elif isinstance(result, list) and len(result) > 0:
        # If it returns a list, get the first item
        mask_item = result[0]
        if isinstance(mask_item, dict) and 'mask' in mask_item:
            mask = mask_item['mask'].convert('L')
        elif hasattr(mask_item, 'convert'):
            mask = mask_item.convert('L')
        else:
            mask = PIL.Image.fromarray((mask_item * 255).astype(np.uint8), mode='L')
    else:
        # Fallback
        mask = PIL.Image.fromarray((result * 255).astype(np.uint8), mode='L')

    image, mask = prepare_img_and_mask(imagex.resize((512, 512)), mask.resize((512, 512)), 'cpu')
    # Run the model
    outputs = rmodel.run(None, {'image': image.numpy().astype(np.float32), 'mask': mask.numpy().astype(np.float32)})

    output = outputs[0][0]
    # Postprocess the outputs
    output = output.transpose(1, 2, 0)
    output = output.astype(np.uint8)
    output = PIL.Image.fromarray(output)
    output = output.resize(size)
    return output.convert('RGBA')


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
    face_mask = face_mask.unsqueeze(0)
    face_mask = face_mask.squeeze().int()*255
    face_mask = face_mask.cpu().numpy()

    return face_mask


def prepareMask(pose_image, face_info,width,height):

    mask = np.zeros([height, width, 3])
    face_mask = pred_face_mask(pose_image, face_info)
    mask[face_mask>0] = 255
    face_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    face_mask = cv2.dilate(face_mask, kernel, iterations=2)
    face_mask = PIL.Image.fromarray(face_mask.astype(np.uint8))

    # (mask, pose, control PIL images), (original positon face + padding: x, y, w, h)
    return face_mask



async def gen_img2img(job_id: str, face_image: PIL.Image.Image, pose_image: PIL.Image.Image, request: Img2ImgRequest):    
    # from bbox crop pose_image
    face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
   
    width, height = pose_image.size
    background = predict(pose_image, (width, height))

    pose_info = pred_info(pose_image)
    face_info = pred_info(face_image)
    mask_img = prepareMask(pose_image, pose_info, width, height)

    control_image = draw_kps(pose_image, pose_info['kps'])
    width, height = pose_image.size
    
    face_embed = np.array(face_info['embedding'])[None, ...]
    id_embeddings = pipeline_swap.get_id_embedding(np.array(face_image))
    image = pipeline_swap.inference(request.prompt, (1, height, width), control_image, face_embed, pose_image, mask_img,
                             request.negative_prompt, id_embeddings, request.ip_adapter_scale, request.guidance_scale, request.num_inference_steps, request.strength)[0]
    
    # Fix: Handle RMBG-2.0 output format for foreground extraction
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = bg_remove_pipe(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    bg_result = pred_pil.resize(image_size)
    bg_result.putalpha(mask)
    
    if isinstance(bg_result, PIL.Image.Image):
        # RMBG-2.0 typically returns the image with background removed directly
        nobackground = bg_result.convert('RGBA')
    elif isinstance(bg_result, list) and len(bg_result) > 0:
        # If it returns a list, handle appropriately
        result_item = bg_result[0]
        if isinstance(result_item, dict):
            if 'image' in result_item:
                nobackground = result_item['image'].convert('RGBA')
            elif 'mask' in result_item:
                # Apply mask to original image
                mask = result_item['mask'].convert('L')
                nobackground = image.copy().convert('RGBA')
                nobackground.putalpha(mask)
            else:
                nobackground = image.convert('RGBA')
        elif hasattr(result_item, 'convert'):
            nobackground = result_item.convert('RGBA')
        else:
            nobackground = image.convert('RGBA')
    else:
        # Fallback to original image
        nobackground = image.convert('RGBA')
    
    new_img = PIL.Image.new("RGBA", (width, height))
    new_img.paste(nobackground, (0, 0), nobackground)
    filename = f"{job_id}_base.png"
    # create new PIL Image has size = top_layer_image
    result_image = PIL.Image.alpha_composite(background, new_img)

    img = cv2.cvtColor(np.array(result_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    face_helper.read_image(img)
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(
        only_center_face=True, resize=640, eye_dist_threshold=5
    )

    face_helper.align_warp_face()
    for cropped_face in face_helper.cropped_faces:
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = codeformer_net(
                    cropped_face_t, w=0.5, adain=True
                )[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f"Failed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)
    bg_img = upsampler.enhance(img, outscale=1)[0]
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(
        upsample_img=bg_img,
        draw_box=False,
        face_upsampler=upsampler,
    )
    # restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    # PIL Image
    filepath = os.path.join(results_dir, filename)
    imwrite(restored_img, str(filepath))
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