import torch
from diffusers import CogVideoXPipeline
from pipeline_stg_cogvideox import CogVideoXSTGPipeline
from diffusers.utils import export_to_video
import os
import re

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

ckpt_path = "THUDM/CogVideoX-2b"
# Load the pipeline
pipe = CogVideoXSTGPipeline.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to("cuda")  # or "THUDM/CogVideoX-2b"

# Define parameters
prompt = (
    "A father and son building a treehouse together, their hands covered in sawdust and smiles on their faces, realistic style."
)

# prompt = re.sub(r'[^a-zA-Z0-9_\- ]', '_', prompt.strip()) # sanitize prompt

#---------Option---------#
stg_mode = "STG"
stg_applied_layers_idx = [11] #0 ~ 41
stg_scale = 1.0 # 0.0 for CFG (default)
do_rescaling = False # False (default)
#------------------------#

guidance_scale = 6
num_inference_steps = 50

pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

# Generate video frames
frames = pipe(
    height=480,
    width=480,
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

# Construct the video filename
if stg_scale == 0:
    video_name = f"CFG_rescale_{do_rescaling}.mp4"
else:
    layers_str = "_".join(map(str, stg_applied_layers_idx))
    video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

# Save video to samples directory
video_path = os.path.join("samples", video_name)
export_to_video(frames, video_path, fps=8)

print(f"Video saved to {video_path}")
