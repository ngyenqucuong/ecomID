import os
from huggingface_hub import hf_hub_download,snapshot_download


if not os.path.exists("./models/antelopev2/"):
    snapshot_download(
        repo_id="InstantX/InstantID", allow_patterns="/models/antelopev2/*", local_dir="./models/antelopev2/"
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
