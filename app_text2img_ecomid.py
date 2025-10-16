from contextlib import asynccontextmanager
import time

import torch
from PIL import Image

from pulid import attention_processor as attention
from pipeline_text2img_ecomid import EcomIDText2ImgPipeline
from insightface.app import FaceAnalysis

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

import logging

from pydantic import BaseModel
import asyncio

from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import json
import uvicorn
import uuid
import cv2
import numpy
import math
import argparse

NSFW_THRESHOLD = 0.85


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = numpy.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = numpy.array(kps)

    w, h = image_pil.size
    out_img = numpy.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(numpy.mean(x)), int(numpy.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(numpy.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(numpy.uint8))
    return out_img_pil


def pred_face_info(img, insightface_app):
    face_info = insightface_app.get(cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR))
    if len(face_info) == 0:
        raise RuntimeError('No face detected in the reference image')
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # only use the maximum face

    return face_info


class EcomIDText2ImgGenerator:
    def __init__(self, face_adapter_path, controlnet_path, base_model_path, device='cuda:0'):
        self.device = device

        # Create args object for pipeline initialization
        class Args:
            pass

        args = Args()
        args.device = device
        args.face_adapter_path = face_adapter_path
        args.controlnet_path = controlnet_path
        args.base_model_path = base_model_path

        # Initialize InsightFace
        self.insightface_app = FaceAnalysis(name='antelopev2', root='./',
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))

        # Initialize pipeline
        self.pipeline = EcomIDText2ImgPipeline(args, self.insightface_app)

        # Set attention processor parameters
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True

        logger.info("EcomID Text2Img Generator initialized successfully")

    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        ref_image,
        width=1024,
        height=1024,
        num_steps=30,
        guidance_scale=5.0,
        id_scale=0.8,
        negative_prompt="flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality",
        seed=None,
    ):
        if seed is None or seed == -1:
            seed = torch.Generator(device="cpu").seed()

        print(f"Generating '{prompt}' with seed {seed}")
        t0 = time.perf_counter()

        # Convert ref_image to PIL if it's numpy array
        if isinstance(ref_image, numpy.ndarray):
            ref_image = Image.fromarray(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))

        # Extract face info from reference image
        face_info = pred_face_info(ref_image, self.insightface_app)
        face_embed = numpy.array(face_info['embedding'])[None, ...]

        # Generate control image from reference face
        kps = face_info['kps']
        control_image = draw_kps(ref_image, kps)

        # Get ID embeddings
        id_embeddings = self.pipeline.get_id_embedding(numpy.array(ref_image))

        # Set random seed
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # Generate image
        image = self.pipeline.inference(
            prompt,
            (1, height, width),
            control_image,
            face_embed,
            negative_prompt,
            id_embeddings,
            id_scale,
            guidance_scale,
            num_steps
        )[0]

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")

        return image, str(seed), self.pipeline.debug_img_list


text2img_generator = None
executor = ThreadPoolExecutor(max_workers=1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_pipelines():
    """Initialize the EcomID Text2Img pipeline - GPU optimized"""
    global text2img_generator
    try:
        # Get checkpoint paths from environment or use defaults
        face_adapter_path = os.getenv('FACE_ADAPTER_PATH', 'checkpoints/ip-adapter.bin')
        controlnet_path = os.getenv('CONTROLNET_PATH', 'checkpoints/ControlNetModel')
        base_model_path = os.getenv('BASE_MODEL_PATH', 'checkpoints/realisticStockPhoto_v20.safetensors')

        text2img_generator = EcomIDText2ImgGenerator(
            face_adapter_path=face_adapter_path,
            controlnet_path=controlnet_path,
            base_model_path=base_model_path,
            device='cuda:0'
        )

        logger.info("Pipelines initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, initialize_pipelines)
    yield


app = FastAPI(title="EcomID Text2Img", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Text2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality"
    seed: Optional[int] = None
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    width: int = 1024
    height: int = 1024
    id_scale: float = 0.8


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


async def gen_text2img(job_id: str, ref_image: Image.Image, request: Text2ImgRequest):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1

        negative_prompt = f"{request.negative_prompt}, bad quality, worst quality, text, signature, watermark, extra limbs"
        seed = request.seed if request.seed is not None else -1

        gen_image, seed, _ = text2img_generator.generate_image(
            prompt=request.prompt,
            ref_image=ref_image,
            width=request.width,
            height=request.height,
            num_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            id_scale=request.id_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )

        jobs[job_id]["progress"] = 0.8

        # Save generated image
        filename = f"{job_id}_text2img.png"
        filepath = os.path.join(results_dir, filename)
        gen_image.save(filepath)

        metadata = {
            "job_id": job_id,
            "type": "text2img",
            "seed": int(seed),
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

        logger.info(f"Text2Img completed successfully on cuda")

    except Exception as e:
        logger.error(f"Error in gen_text2img: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["progress"] = 0.0


@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    try:
        with open("interface_text2img.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1><p>Please create interface_text2img.html</p>")


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

    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "pipelines_loaded": text2img_generator is not None,
        "gpu_info": gpu_info
    }


@app.post("/text2img")
async def text2img(
    ref_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(5.0),
    id_scale: float = Form(0.8),
    seed: Optional[int] = Form(-1),
    width: int = Form(1024),
    height: int = Form(1024),
):
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "text2img"
    }
    logger.info(f"Received text2img request with job_id: {job_id}")

    try:
        # Load reference image
        ref_img_bytes = await ref_image.read()
        ref_img = cv2.imdecode(numpy.frombuffer(ref_img_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        ref_img_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))

        request = Text2ImgRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            id_scale=id_scale,
        )

        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_text2img(job_id, ref_img_pil, request)
        ))

        return {"job_id": job_id, "status": "pending"}

    except Exception as e:
        logger.error(f"Error processing text2img request: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}


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
    # Parse command line arguments for checkpoint paths
    parser = argparse.ArgumentParser(description='EcomID Text2Img FastAPI Server')
    parser.add_argument('--face_adapter_path', default='checkpoints/ip-adapter.bin', help='Path for ip-adapter')
    parser.add_argument('--controlnet_path', default='checkpoints/ControlNetModel', help='Path for controlnet model')
    parser.add_argument('--base_model_path', default='checkpoints/realisticStockPhoto_v20.safetensors', help='Path for SDXL model')
    parser.add_argument('--port', type=int, default=8889, help='Port to run the server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')

    args = parser.parse_args()

    # Set environment variables for checkpoint paths
    os.environ['FACE_ADAPTER_PATH'] = args.face_adapter_path
    os.environ['CONTROLNET_PATH'] = args.controlnet_path
    os.environ['BASE_MODEL_PATH'] = args.base_model_path

    uvicorn.run(app, host=args.host, port=args.port)
