import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from pipeline_stg_hunyuan import HunyuanVideoSTGPipeline
from diffusers.utils import export_to_video
import re
import os

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

model_id = "/scratch/x2927a30/checkpoint/hunyuanvideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
)
pipe = HunyuanVideoSTGPipeline.from_pretrained(model_id, transformer=transformer, revision='refs/pr/18', torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

#--------Option--------#
stg_mode = "STG-R"
stg_applied_layers_idx = [i for i in range(20, 40, 2)]
stg_scales = [1.0, 0.0] # 0.0 for CFG (default)
do_rescaling = False
#----------------------#

with open("prompts/prompt_cfg_bad.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

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
                prompt=prompt,
                height=320,
                width=512,
                num_frames=61,
                num_inference_steps=30,
                stg_mode=stg_mode,
                stg_applied_layers_idx=[layer_idx],
                stg_scale=stg_scale,
                do_rescaling=do_rescaling,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            
            # Save video to mode-specific directory
            export_to_video(frames, video_path, fps=15)
            
            print(f"Video saved to {video_path}")