import torch
from diffusers import CogVideoXPipeline
from pipeline_cogvideox_stg import CogVideoXSTGPipeline
from diffusers.utils import export_to_video
import os
import re

def sanitize_and_truncate(prompt, limit=20):
    # Step 1: Remove special characters (keeping basic punctuation)
    sanitized = re.sub(r'["\',]', '', prompt)
    sanitized = re.sub(r'[^\w\s.!?\-]', '', sanitized)

    # Step 2: Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Step 3: Split into words and truncate to the first `limit` words
    words = sanitized.split()
    truncated = ' '.join(words[:limit])

    return truncated

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

ckpt_path = "/scratch/x2927a30/checkpoint/cogvideox"
# Load the pipeline
pipe = CogVideoXSTGPipeline.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to("cuda")  # or "THUDM/CogVideoX-2b"

# prompt = re.sub(r'[^a-zA-Z0-9_\- ]', '_', prompt.strip()) # sanitize prompt

#---------Option---------#
stg_mode = "STG-R"
stg_applied_layers_idx = [i for i in range(42)]
stg_scales = [1.0, 0.0] # 0.0 for CFG (default)
do_rescaling = False # False (default)
#------------------------#

height = 480
width = 480
guidance_scale = 6
num_inference_steps = 50

pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

with open("prompts/prompt1.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# with open("prompts/prompt2.txt", "r") as f:
#     prompts += [line.strip() for line in f.readlines() if line.strip()]

# with open("prompts/prompt3.txt", "r") as f:
#     prompts += [line.strip() for line in f.readlines() if line.strip()]

# with open("prompts/prompt_stg.txt", "r") as f:
#     prompts = [line.strip() for line in f.readlines() if line.strip()]

for i, prompt in enumerate(prompts):
    sanitized_prompt = sanitize_and_truncate(prompt)
    for layer_idx in stg_applied_layers_idx:
        for stg_scale in stg_scales:
            if stg_scale > 0:
                mode = stg_mode
            else:
                mode = "CFG"
            mode_dir = os.path.join("data", mode, f"scale_{stg_scale}_layer_{layer_idx}")
            if mode in ["CFG", "cond"]:
                mode_dir = os.path.join("data", mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            # Construct the video filename
            video_name = f"{sanitized_prompt}.mp4"
            video_path = os.path.join(mode_dir, video_name)
                
            # Skip if video already exists
            if os.path.exists(video_path):
                print(f"Video already exists: {video_path}, skipping...")
                continue
            # Generate video frames
            frames = pipe(
                prompt, 
                height=height,
                width=width,
                num_frames=31,
                stg_mode=stg_mode,
                stg_applied_layers_idx=[layer_idx],
                stg_scale=stg_scale,
                do_rescaling=do_rescaling,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
            
            # Save video to mode-specific directory
            export_to_video(frames, video_path, fps=8)
            
            print(f"Video saved to {video_path}")