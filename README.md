# FaceSwap
Facilitate the exchange of facial features, including hair characteristics, between two photographs with high identity fidelity, utilizing the EcomID framework.



## Get Started
```sh
python face_swap_ecomid.py  
    --src images/kaifu_resize.png
    --dst images/musk_resize.jpeg 
    --outdir results
    --face_adapter_path checkpoints/ip-adapter.bin 
    --controlnet_path  checkpoints/ControlNetModel 
    --base_model_path checkpoints/realisticStockPhoto_v20.safetensors
```

## Example
| Destination| Source | Result |
| --- | --- | --- |
| ![](images/musk_resize.jpeg) |![](images/kaifu_resize.png) | ![](images/musk_resize_swaped.png) |
| ![](images/yann-lecun_resize.jpg) |![](images/kaifu_resize.png) | ![](images/yann-lecun_resize_swaped.png) |
| ![](images/test.jpg) |![](images/kaifu_resize.png) | ![](images/test_swaped.png) |
| ![](images/Cook.jpg) |![](images/kaifu_resize.png) | ![](images/Cook_swaped.png) |
## Model Zoo

|                                              Version                                               | Base Model |                                                                                              Description                                                                                              |
|:--------------------------------------------------------------------------------------------------:|:----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|              [realisticStockPhoto_v20](https://huggingface.co/lllyasviel/fav_models/blob/main/fav/realisticStockPhoto_v20.safetensors)              |    SDXL    |     base model          |
| [face_adapter](https://huggingface.co/InstantX/InstantID/tree/main) |    InstantID    |            face embedding              |
| [Controlnet](https://huggingface.co/InstantX/InstantID/tree/main) |    InstantID    | landmarks controlnet |


## References
https://github.com/alimama-creative/SDXL_EcomID_ComfyUI \
https://github.com/ToTheBeginning/PuLID \
https://github.com/instantX-research/InstantID \
https://github.com/huggingface/diffusers
