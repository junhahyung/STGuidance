# 🚀[CVPR 2025] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling✨

## 📑Paper
- Arxiv: [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664)

## 🌐Project Page
- [STG Project Page](https://junhahyung.github.io/STGuidance)

## 📰 News
- **[2025.03.08]** 🚀 **STG is now integrated into the Diffusers community pipeline!**  
  👉 [**Check it out on Hugging Face**](https://github.com/huggingface/diffusers/tree/main/examples/community)

- **[2025.02.07]** 🏆 **STG officially accepted to CVPR 2025!**  
  🎤 Stay tuned for our presentation at the conference.

- **[2024.12.20]** 🔥 **STG added to LTXVideo’s official repository!**  
  📂 Now part of [**LTXVideo’s main repository**](https://github.com/Lightricks/ComfyUI-LTXVideo).

- **[2024.12.19]** 🖥️ **ComfyUI STG support for LTXVideo!**  
  🎬 Implemented in [**ComfyUI**](https://github.com/Lightricks/ComfyUI-LTXVideo), enhancing **LTXVideo** support.

## 🎥Video Examples
Below are example videos showcasing the enhanced video quality achieved through STG:

### Mochi


https://github.com/user-attachments/assets/b8795d10-b7dd-4928-84b0-1335fac1af03




https://github.com/user-attachments/assets/7eb5391c-f655-4e42-b704-df9b6125dea1

### HunyuanVideo


https://github.com/user-attachments/assets/3ccd4a63-15e6-4473-b693-8b757b3ae6b1



https://github.com/user-attachments/assets/492f43d0-c1bd-4941-b90b-8fe3d22a2e6b




### CogVideoX


https://github.com/user-attachments/assets/adc5af40-e50d-4b00-b98b-8e88ee04bae8


https://github.com/user-attachments/assets/fcb8a078-58a5-4e62-a55e-662a0b08216b


### SVD (Stable Video Diffusion)



https://github.com/user-attachments/assets/5d11b8dc-e63d-4ac9-80d8-c81735fcf181



https://github.com/user-attachments/assets/29afec1b-f137-48d4-b237-e2058431ccee


### LTX-Video


https://github.com/user-attachments/assets/4cd722cd-c6e8-428d-8183-65e5954a930b





## 🗺️Start Guide
🧪**Diffusers-based codes**
   To run the test script, refer to the `inference.py` file in each folder. Below is an example using Mochi:
   
   ```python
   # inference.py
   import torch
   from diffusers import MochiPipeline
   from pipeline_stg_mochi import MochiSTGPipeline
   from diffusers.utils import export_to_video
   import os
   
   # Ensure the samples directory exists
   os.makedirs("samples", exist_ok=True)
   
   ckpt_path = "genmo/mochi-1-preview"
   # Load the pipeline
   pipe = MochiSTGPipeline.from_pretrained(ckpt_path, variant="bf16", torch_dtype=torch.bfloat16)
   
   # Enable memory savings
   # pipe.enable_model_cpu_offload()
   # pipe.enable_vae_tiling()
   pipe = pipe.to("cuda")
   
   #--------Option--------#
   prompt = "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."
   stg_applied_layers_idx = [34]
   stg_mode = "STG"
   stg_scale = 1.0 # 0.0 for CFG (default)
   do_rescaling = False # False (default)
   #----------------------#
   
   # Generate video frames
   frames = pipe(
       prompt, 
       height=480,
       width=480,
       num_frames=81,
       stg_applied_layers_idx=stg_applied_layers_idx,
       stg_scale=stg_scale,
       generator = torch.Generator().manual_seed(42),
       do_rescaling=do_rescaling,
   ).frames[0]
   
   # Construct the video filename
   if stg_scale == 0:
       video_name = f"CFG_rescale_{do_rescaling}.mp4"
   else:
       layers_str = "_".join(map(str, stg_applied_layers_idx))
       video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"
   
   # Save video to samples directory
   video_path = os.path.join("samples", video_name)
   export_to_video(frames, video_path, fps=30)
   
   print(f"Video saved to {video_path}")
   ```
   For details on memory efficiency, inference acceleration, and more, refer to the original pages below:
   - [Mochi](https://huggingface.co/genmo/mochi-1-preview)
   - [CogVideoX](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
   - [HunyuanVideo](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video)
   - [StableVideoDiffusion](https://huggingface.co/docs/diffusers/en/using-diffusers/svd)


## 🙏Acknowledgements
This project is built upon the following works:
- [Mochi](https://github.com/genmoai/mochi?tab=readme-ov-file)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [LTX-Video](https://github.com/Lightricks/LTX-Video)
- [diffusers](https://github.com/huggingface/diffusers)

