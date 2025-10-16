# FaceSwap with EcomID

Facilitate the exchange of facial features, including hair characteristics, between two photographs with high identity fidelity, utilizing the EcomID framework.

This project supports two methods:
1. **Face Swap (Inpainting)**: Swap faces between two images using inpainting
2. **Text2Img with Reference Face**: Generate new images from text prompts while maintaining face identity from a reference image

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Method 1: Face Swap (Inpainting)](#method-1-face-swap-inpainting)
  - [Method 2: Text2Img with Reference Face](#method-2-text2img-with-reference-face)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Model Zoo](#model-zoo)
- [Examples](#examples)
- [References](#references)

---

## Features

### Face Swap (Inpainting)
- High-fidelity face swapping between two images
- Preserves hair characteristics from source image
- Uses EcomID + InstantID framework
- Mask-based inpainting for natural results

### Text2Img with Reference Face
- Generate entirely new images from text prompts
- Maintains facial identity from reference image
- Full control over image composition via text
- FastAPI server with web interface
- RESTful API for integration
- Asynchronous job processing
- Real-time progress tracking

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 12GB+ VRAM)
- CUDA 11.7+ and cuDNN

### Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install insightface onnxruntime-gpu
pip install opencv-python pillow numpy
pip install fastapi uvicorn python-multipart  # For API server
pip install facexlib
pip install safetensors
```

### Model Checkpoints
Download the required model checkpoints:

1. **Base Model** (SDXL):
   - [realisticStockPhoto_v20.safetensors](https://huggingface.co/lllyasviel/fav_models/blob/main/fav/realisticStockPhoto_v20.safetensors)
   - Place in: `checkpoints/realisticStockPhoto_v20.safetensors`

2. **InstantID Models**:
   - Download from [InstantX/InstantID](https://huggingface.co/InstantX/InstantID/tree/main)
   - `ip-adapter.bin` → `checkpoints/ip-adapter.bin`
   - `ControlNetModel/` → `checkpoints/ControlNetModel/`

3. **PuLID Model** (auto-downloaded on first run):
   - Downloaded from [guozinan/PuLID](https://huggingface.co/guozinan/PuLID)
   - Saved to: `models/pulid_v1.1.safetensors`

4. **Face Analysis Model** (auto-downloaded on first run):
   - AntelopeV2 from InsightFace
   - Saved to: `models/antelopev2/`

---

## Quick Start

### Method 1: Face Swap (Inpainting)

Swap faces between a source image and destination image.

```bash
python face_swap_ecomid.py \
    --src images/source_face.png \
    --dst images/destination_image.jpeg \
    --outdir results \
    --face_adapter_path checkpoints/ip-adapter.bin \
    --controlnet_path checkpoints/ControlNetModel \
    --base_model_path checkpoints/realisticStockPhoto_v20.safetensors
```

**Parameters:**
- `--src`: Source image (face to extract)
- `--dst`: Destination image (face to replace)
- `--outdir`: Output directory for results
- `--face_adapter_path`: Path to IP-Adapter weights
- `--controlnet_path`: Path to ControlNet model
- `--base_model_path`: Path to SDXL base model
- `--device`: Device to use (default: `cuda:0`)

---

### Method 2: Text2Img with Reference Face

Generate new images from text prompts while preserving facial identity.

#### Option A: Command Line Interface

```bash
python face_text2img_ecomid.py \
    --ref_image images/reference_face.png \
    --prompt "a person in a suit, professional photo, high quality" \
    --outdir results \
    --face_adapter_path checkpoints/ip-adapter.bin \
    --controlnet_path checkpoints/ControlNetModel \
    --base_model_path checkpoints/realisticStockPhoto_v20.safetensors \
    --width 1024 \
    --height 1024 \
    --steps 30 \
    --guidance_scale 5.0 \
    --id_scale 0.8
```

**Parameters:**
- `--ref_image`: Reference face image
- `--prompt`: Text prompt for generation
- `--outdir`: Output directory
- `--width`: Output width (default: 1024)
- `--height`: Output height (default: 1024)
- `--steps`: Inference steps (default: 30)
- `--guidance_scale`: CFG scale (default: 5.0)
- `--id_scale`: Identity preservation strength (default: 0.8)

#### Option B: FastAPI Server (Recommended)

**Start the server:**
```bash
python app_text2img_ecomid.py \
    --face_adapter_path checkpoints/ip-adapter.bin \
    --controlnet_path checkpoints/ControlNetModel \
    --base_model_path checkpoints/realisticStockPhoto_v20.safetensors \
    --port 8889 \
    --host 0.0.0.0
```

**Access the web interface:**
Open your browser and navigate to: `http://localhost:8889`

**Server Features:**
- Beautiful web interface with real-time preview
- Asynchronous job processing (non-blocking)
- Progress tracking with status updates
- Job management (list, delete, download)
- RESTful API for integration
- Metadata saving for reproducibility

---

## Project Structure

```
FaceSwap/
├── face_swap_ecomid.py                          # CLI for face swapping (inpainting)
├── face_text2img_ecomid.py                      # CLI for text2img generation
├── app_text2img_ecomid.py                       # FastAPI server for text2img
├── interface_text2img.html                      # Web interface for text2img
│
├── pipeline_ecomid.py                           # EcomID pipeline wrapper (inpainting)
├── pipeline_text2img_ecomid.py                  # EcomID pipeline wrapper (text2img)
├── pipeline_stable_diffusion_xl_inpaint_ecomid.py  # SDXL inpainting pipeline
├── pipeline_stable_diffusion_xl_text2img_ecomid.py # SDXL text2img pipeline
│
├── pulid/                                       # PuLID implementation
│   ├── attention_processor.py                   # Custom attention processors
│   ├── encoders_transformer.py                  # ID encoder (IDFormer)
│   └── utils.py                                 # Utility functions
│
├── ip_adapter/                                  # IP-Adapter implementation
│   ├── attention_processor.py                   # IP-Adapter attention
│   └── resampler.py                             # Image projection model
│
├── eva_clip/                                    # EVA-CLIP for face features
├── trainscripts/                                # Training utilities
│
├── checkpoints/                                 # Model checkpoints (user provided)
│   ├── ip-adapter.bin
│   ├── ControlNetModel/
│   └── realisticStockPhoto_v20.safetensors
│
├── models/                                      # Auto-downloaded models
│   ├── pulid_v1.1.safetensors
│   └── antelopev2/
│
├── results/                                     # Output images and metadata
└── images/                                      # Sample images
```

---

## API Documentation

### FastAPI Server Endpoints

#### POST `/text2img`
Generate an image from text prompt with reference face.

**Request (multipart/form-data):**
```
ref_image: File (image file)
prompt: str
negative_prompt: str (optional)
width: int (default: 1024)
height: int (default: 1024)
num_inference_steps: int (default: 30)
guidance_scale: float (default: 5.0)
id_scale: float (default: 0.8)
seed: int (default: -1, random)
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending"
}
```

#### GET `/job/{job_id}`
Check job status and progress.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 1.0,
  "result_url": "/results/uuid_text2img.png",
  "seed": 12345,
  "created_at": "2025-01-01T12:00:00",
  "completed_at": "2025-01-01T12:01:30"
}
```

#### GET `/results/{filename}`
Download generated image.

#### GET `/jobs`
List all jobs (sorted by creation time, descending).

#### DELETE `/job/{job_id}`
Delete a job and its associated files.

#### GET `/health`
Health check with GPU information.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "pipelines_loaded": true,
  "gpu_info": {
    "gpu_name": "NVIDIA RTX 4090",
    "gpu_memory_total": 25769803776,
    "gpu_memory_allocated": 12884901888,
    "gpu_memory_cached": 13958643712
  }
}
```

### API Example (Python)

```python
import requests

# Submit job
with open("reference.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8889/text2img",
        files={"ref_image": f},
        data={
            "prompt": "a person in a suit, professional photo",
            "width": 1024,
            "height": 1024,
            "guidance_scale": 5.0,
            "id_scale": 0.8,
        }
    )

job_id = response.json()["job_id"]

# Check status
import time
while True:
    status = requests.get(f"http://localhost:8889/job/{job_id}").json()
    print(f"Status: {status['status']}, Progress: {status['progress']}")

    if status["status"] == "completed":
        # Download result
        result_url = status["result_url"]
        img = requests.get(f"http://localhost:8889{result_url}")
        with open("result.png", "wb") as f:
            f.write(img.content)
        break

    time.sleep(1)
```

---

## Configuration

### Key Parameters

#### For Face Swap (Inpainting)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `id_scale` | 0.5 | Identity preservation strength (0-2) |
| `steps` | 30 | Number of denoising steps |
| `strength` | 0.9 | Inpainting strength (0-1) |
| `guidance_scale` | 3.5 | Classifier-free guidance scale |

#### For Text2Img
| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 1024 | Output image width |
| `height` | 1024 | Output image height |
| `steps` | 30 | Number of denoising steps |
| `guidance_scale` | 5.0 | Classifier-free guidance scale |
| `id_scale` | 0.8 | Identity preservation strength (0-2) |
| `seed` | -1 | Random seed (-1 for random) |

### Attention Processor Settings

These are configured in the code:
```python
attention.NUM_ZERO = 8      # Number of zero attention layers
attention.ORTHO = False     # Orthogonal attention
attention.ORTHO_v2 = True   # Orthogonal attention v2
```

---

## Model Zoo

| Model | Base | Description | Download |
|-------|------|-------------|----------|
| realisticStockPhoto_v20 | SDXL | Realistic photo generation | [Link](https://huggingface.co/lllyasviel/fav_models/blob/main/fav/realisticStockPhoto_v20.safetensors) |
| IP-Adapter (InstantID) | InstantID | Face embedding projection | [Link](https://huggingface.co/InstantX/InstantID/tree/main) |
| ControlNet (InstantID) | InstantID | Facial landmark control | [Link](https://huggingface.co/InstantX/InstantID/tree/main) |
| PuLID v1.1 | - | Identity encoder | [Link](https://huggingface.co/guozinan/PuLID) |
| AntelopeV2 | InsightFace | Face detection & analysis | Auto-download |
| EVA-CLIP-L-14-336 | EVA-CLIP | Face feature extraction | Auto-download |

---

## Examples

### Face Swap (Inpainting) Examples

| Destination | Source | Result |
|-------------|--------|--------|
| ![](images/musk_resize.jpeg) | ![](images/kaifu_resize.png) | ![](images/musk_resize_swaped.png) |
| ![](images/yann-lecun_resize.jpg) | ![](images/kaifu_resize.png) | ![](images/yann-lecun_resize_swaped.png) |
| ![](images/test.jpg) | ![](images/kaifu_resize.png) | ![](images/test_swaped.png) |
| ![](images/Cook.jpg) | ![](images/kaifu_resize.png) | ![](images/Cook_swaped.png) |

### Text2Img with Reference Face

Generate completely new images from text prompts while maintaining facial identity from a reference image.

**Example Prompts:**
- "a person in a suit, professional photo, high quality"
- "a person at the beach, sunset, cinematic lighting"
- "a person as a superhero, comic book style"
- "portrait photo, studio lighting, professional"

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce image resolution (`--width 512 --height 512`)
- Reduce inference steps (`--steps 20`)
- Close other GPU applications

**2. No Face Detected**
- Ensure reference image has a clear, visible face
- Try a different reference image with better face visibility
- Check image is not corrupted

**3. Low Identity Preservation**
- Increase `id_scale` (try 1.0 or higher)
- Use higher quality reference images
- Ensure good lighting in reference image

**4. Model Not Found**
- Check checkpoint paths are correct
- Verify files are downloaded to correct directories
- Check file permissions

**5. Server Won't Start**
- Check if port is already in use
- Verify all dependencies are installed
- Check GPU availability with `nvidia-smi`

---

## Development Notes

### Recent Changes (Current Session)
- Added Text2Img pipeline with reference face support
- Created FastAPI server (`app_text2img_ecomid.py`)
- Added web interface (`interface_text2img.html`)
- Removed background removal step from text2img pipeline
- Implemented job queue system with progress tracking

### Pipeline Architecture

**Inpainting Method:**
```
Source Image → Face Detection → Face Embedding (InsightFace + EVA-CLIP)
                                      ↓
Destination → Mask Generation → SDXL Inpaint + ControlNet → Result
```

**Text2Img Method:**
```
Reference Face → Face Detection → Face Embedding (PuLID + InstantID)
                                        ↓
Text Prompt → SDXL Text2Img + ControlNet → Generated Image
```

---

## References

- [SDXL EcomID ComfyUI](https://github.com/alimama-creative/SDXL_EcomID_ComfyUI)
- [PuLID](https://github.com/ToTheBeginning/PuLID)
- [InstantID](https://github.com/instantX-research/InstantID)
- [Diffusers](https://github.com/huggingface/diffusers)
- [InsightFace](https://github.com/deepinsight/insightface)
- [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)

---

## License

Please refer to individual model licenses for usage restrictions.

---

## Citation

If you use this code in your research, please cite the relevant papers:
- EcomID
- PuLID
- InstantID
- Stable Diffusion XL

---

**Last Updated:** January 2025
