import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from pipeline import WanSTGPipeline

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "/home/nas4_user/kinamkim/checkpoint/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanSTGPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=480,
        num_frames=49,
        guidance_scale=5.0,
        skip_guidance_scale=2.0,
        skip_guidance_block_idxs=[2], # 30
        ).frames[0]
export_to_video(output, "output.mp4", fps=16)