import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from pipeline_stg_hunyuan import HunyuanVideoSTGPipeline
from diffusers.utils import export_to_video

model_id = "/scratch/x2927a30/checkpoint/hunyuanvideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
)
pipe = HunyuanVideoSTGPipeline.from_pretrained(model_id, transformer=transformer, revision='refs/pr/18', torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

#--------Option--------#
stg_mode = "STG-R"
stg_applied_layers_idx = [2]
stg_scale = 1.0
do_rescaling = False
#----------------------#

output = pipe(
    prompt="A wolf howling at the moon, with the moon subtly resembling a giant clock face, realistic style.",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
    stg_mode=stg_mode,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling,
    generator=torch.Generator(device="cuda").manual_seed(42),
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
export_to_video(video, video_path, fps=15)

print(f"Video saved to {video_path}")
