import os
from huggingface_hub import hf_hub_download
from basicsr.utils.download_util import load_file_from_url


if not os.path.exists("./models/antelopev2/"):
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/1k3d68.onnx",
    local_dir="./",
    repo_type="space"
    )   
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/2d106det.onnx",
    local_dir="./",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/genderage.onnx",
    local_dir="./",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/glintr100.onnx",
    local_dir="./",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/scrfd_10g_bnkps.onnx",
    local_dir="./",
    repo_type="space"
    )
    # run 'mv models/antelopev2/antelopev2/* models/antelopev2/' cmd

if not os.path.exists("./checkpoints/"):
    # make dir 'ControlNetModel'
    os.makedirs("./checkpoints/ControlNetModel", exist_ok=True)
    hf_hub_download(
        repo_id="InstantX/InstantID",
        subfolder="ControlNetModel",
        filename="config.json",
        local_dir="./checkpoints"
    )
    
    hf_hub_download(
        repo_id="InstantX/InstantID",
        subfolder="ControlNetModel",
        filename="diffusion_pytorch_model.safetensors",
        local_dir="./checkpoints"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="./checkpoints"
    )

os.system("wget https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
}
# download weights
if not os.path.exists('CodeFormer/weights/CodeFormer/codeformer.pth'):
    load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='CodeFormer/weights/CodeFormer', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
    load_file_from_url(url=pretrain_model_url['detection'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/parsing_parsenet.pth'):
    load_file_from_url(url=pretrain_model_url['parsing'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir='CodeFormer/weights/realesrgan', progress=True, file_name=None)