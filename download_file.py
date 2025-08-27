import os
from huggingface_hub import hf_hub_download


if not os.path.exists("./models/antelopev2/"):
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/1k3d68.onnx",
    repo_type="space"
    )   
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/2d106det.onnx",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/genderage.onnx",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/glintr100.onnx",
    repo_type="space"
    )
    hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="models/antelopev2/scrfd_10g_bnkps.onnx",
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

