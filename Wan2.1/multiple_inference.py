import os
import torch
import re
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from pipeline import WanSTGPipeline

#-------Option--------#
# Model options
model_id = "/home/nas4_user/kinamkim/checkpoint/Wan2.1-T2V-1.3B-Diffusers"
prompt_file = "prompts.txt"  # Path to a file containing prompts (one per line)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# Generation options
height = 480
width = 720
num_frames = 49
guidance_scale = 5.0
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
fps = 16
#-----------------------#

def sanitize_filename(text):
    """Convert text to a valid filename by removing or replacing invalid characters."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", text)
    # Limit length to avoid too long filenames
    return sanitized[:100]

def main():
    # Load model and setup pipeline
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanSTGPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cuda")

    # Read prompts from file
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Process each prompt
    for prompt in prompts:
        print(f"Processing prompt: {prompt}")
        sanitized_prompt = sanitize_filename(prompt)
        
        # Create scales to iterate through
        scales = [0.0, 1.0, 2.0, 3.0]
        
        for scale in scales:
            if scale == 0.0:
                # For scale 0.0, use CFG only (no STG)
                output_dir = f"outputs/{sanitized_prompt}/CFG"
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/output.mp4"
                if os.path.exists(output_path):
                    print(f"Skipping {output_path} because it already exists")
                    continue
                
                print(f"Generating with CFG only (scale={scale})")
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    skip_guidance_scale=0.0,  # No STG
                    skip_guidance_block_idxs=[],  # No blocks
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).frames[0]
                
                export_to_video(output, output_path, fps=fps)
                print(f"Saved to {output_path}")
            else:
                # For other scales, iterate through block indices
                for block_idx in range(30):  # 0 to 29
                    output_dir = f"outputs/{sanitized_prompt}/STG/scale_{scale}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = f"{output_dir}/block_{block_idx}.mp4"
                    if os.path.exists(output_path):
                        print(f"Skipping {output_path} because it already exists")
                        continue
                    
                    print(f"Generating with STG: scale={scale}, block_idx={block_idx}")
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        skip_guidance_scale=scale,
                        skip_guidance_block_idxs=[block_idx],
                        generator=torch.Generator(device="cuda").manual_seed(42),
                    ).frames[0]
                    
                    export_to_video(output, output_path, fps=fps)
                    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main() 