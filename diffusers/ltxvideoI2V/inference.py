import os
import torch
from diffusers import LTXPipeline
from pipeline_stg_ltx_image2video import LTXImageToVideoSTGPipeline
from diffusers.utils import export_to_video, load_image

ckpt_path = "Lightricks/LTX-Video"
pipe = LTXImageToVideoSTGPipeline.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/11.png"
)

prompt = "A medieval fantasy scene featuring a rugged man with shoulder-length brown hair and a beard. He wears a dark leather tunic over a maroon shirt with intricate metal details. His facial expression is serious and intense, and he is making a gesture with his right hand, forming a small circle with his thumb and index finger. The warm golden lighting casts dramatic shadows on his face. The background includes an ornate stone arch and blurred medieval-style decor, creating an epic atmosphere."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

stg_mode = "STG-A" # STG-A, STG-R
stg_applied_layers_idx = [19] # 0~27
stg_scale = 0.0 # 0.0 for CFG
do_rescaling = True # Default (False)

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=480,
    height=480,
    num_frames=81,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
    stg_mode=stg_mode,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling
).frames[0]

if stg_scale == 0:
    video_name = f"CFG_rescale_{do_rescaling}.mp4"
else:
    layers_str = "_".join(map(str, stg_applied_layers_idx))
    video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

# Save video to samples directory
sample_dir = "samples"
os.makedirs(sample_dir, exist_ok=True)
video_path = os.path.join(sample_dir, video_name)
export_to_video(video, video_path, fps=24)

print(f"Video saved to {video_path}")
