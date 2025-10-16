import cv2
import torch
import numpy as np
from PIL import Image
from pulid import attention_processor as attention
from pipeline_text2img_ecomid import EcomIDText2ImgPipeline
from insightface.app import FaceAnalysis
import math
import PIL.Image
import argparse

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


def pred_face_info(img, insightface_app):
    face_info = insightface_app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # only use the maximum face

    return face_info


import os
import random

random_seed = 0
generator = torch.Generator(device="cuda")
generator.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FaceText2ImgApp')
    parser.add_argument('--ref_image', required=True, help='Path for reference face image')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--outdir', required=True, help='Path for storing output images')
    parser.add_argument('--face_adapter_path', required=True, help='Path for ipadpter')
    parser.add_argument('--controlnet_path', required=True, help='Path for controlnet model path')
    parser.add_argument('--base_model_path', required=True, help='Path for sdxl model path')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--width', type=int, default=1024, help='Output image width')
    parser.add_argument('--height', type=int, default=1024, help='Output image height')
    parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--id_scale', type=float, default=0.8, help='Identity scale')

    args = parser.parse_args()

    device = torch.device(args.device)
    insightface_app = FaceAnalysis(name='antelopev2', root='./',
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))

    pipeline_text2img = EcomIDText2ImgPipeline(args, insightface_app)

    # other params
    attention.NUM_ZERO = 8
    attention.ORTHO = False
    attention.ORTHO_v2 = True

    # Infer setting
    prompt = args.prompt
    n_prompt = "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality"

    width = args.width
    height = args.height

    # Load reference face image and extract embeddings
    face_image = Image.open(args.ref_image).convert('RGB')
    face_info = pred_face_info(face_image, insightface_app)
    face_embed = np.array(face_info['embedding'])[None, ...]

    # Generate control image from reference face
    kps = face_info['kps']
    control_image = draw_kps(face_image, kps)

    id_embeddings = pipeline_text2img.get_id_embedding(np.array(face_image))

    id_scale = args.id_scale
    steps = args.steps
    guidance_scale = args.guidance_scale

    os.makedirs(args.outdir, exist_ok=True)

    # Generate image using text2img with reference face
    image = pipeline_text2img.inference(
        prompt,
        (1, height, width),
        control_image,
        face_embed,
        n_prompt,
        id_embeddings,
        id_scale,
        guidance_scale,
        steps
    )[0]

    output_name = os.path.basename(args.ref_image).split('.')[0]
    image.save(os.path.join(args.outdir, output_name + f"_text2img.png"))
    print(f"Generated image saved to: {os.path.join(args.outdir, output_name + '_text2img.png')}")
