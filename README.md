# 🚀Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling✨

## 📑Paper
- Arxiv: [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664)

## 🌐Project Page
- [STG Project Page](https://junhahyung.github.io/STGuidance)

## 🎥Video Examples
Below are example videos showcasing the enhanced video quality achieved through STG:

### Mochi


https://github.com/user-attachments/assets/b8795d10-b7dd-4928-84b0-1335fac1af03




https://github.com/user-attachments/assets/7eb5391c-f655-4e42-b704-df9b6125dea1



### CogVideoX


https://github.com/user-attachments/assets/adc5af40-e50d-4b00-b98b-8e88ee04bae8


https://github.com/user-attachments/assets/fcb8a078-58a5-4e62-a55e-662a0b08216b


### SVD (Stable Video Diffusion)



https://github.com/user-attachments/assets/5d11b8dc-e63d-4ac9-80d8-c81735fcf181



https://github.com/user-attachments/assets/29afec1b-f137-48d4-b237-e2058431ccee


### HunyuanVideo


https://github.com/user-attachments/assets/3ccd4a63-15e6-4473-b693-8b757b3ae6b1



https://github.com/user-attachments/assets/492f43d0-c1bd-4941-b90b-8fe3d22a2e6b



### LTX-Video


https://github.com/user-attachments/assets/4cd722cd-c6e8-428d-8183-65e5954a930b





## 🗺️Start Guide
1. 🍡**Mochi**
   - For installation and requirements, refer to the [official repository](https://github.com/genmoai/mochi).
     
   - Update `demos/config.py` with your desired settings and simply run:
     ```bash
     python ./demos/cli.py
     ```

2. 🌌**HunyuanVideo**
   - For installation and requirements, refer to the [official repository](https://github.com/Tencent/HunyuanVideo).
     
   **Using CFG (Default Model):**
   ```bash
   torchrun --nproc_per_node=4 sample_video.py \
    --video-size 544 960 \
    --video-length 65 \
    --infer-steps 50 \
    --prompt "A time traveler steps out of a glowing portal into a Victorian-era street filled with horse-drawn carriages, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --save-path ./results
   ```

   **To utilize STG, use the following command:**
   ```bash
   torchrun --nproc_per_node=4 sample_video.py \
    --video-size 544 960 \
    --video-length 65 \
    --infer-steps 50 \
    --prompt "A time traveler steps out of a glowing portal into a Victorian-era street filled with horse-drawn carriages, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 4 \
    --ring-degree 1 \
    --save-path ./results \
    --stg-mode "STG-R" \
    --stg-block-idx 2 \
    --stg-scale 2.0
   ```
   Key Parameters:
   - **stg_mode**: Only STG-R supported.
   - **stg_scale**: 2.0 is recommended.
   - **stg_block_idx**: Specify the block index for applying STG.

3. 🏎️**LTX-Video**
   - For installation and requirements, refer to the [official repository](https://github.com/Lightricks/LTX-Video).

   **Using CFG (Default Model):**
   ```bash
   python inference.py --ckpt_dir './weights' --prompt "A man ..."
   ```

   **To utilize STG, use the following command:**
   ```bash
   python inference.py --ckpt_dir './weights' --prompt "A man ..." --stg_mode stg-a --stg_scale 1.0 --stg_block_idx 19 --do_rescaling True
   ```
   Key Parameters:
   - **stg_mode**: Choose between stg-a or stg-r.
   - **stg_scale**: Recommended values are ≤2.0.
   - **stg_block_idx**: Specify the block index for applying STG.
   - **do_rescaling**: Set to True to enable rescaling.
     
4. 🧪**Diffusers**
   
   The [Diffusers implementation](https://github.com/junhahyung/STGuidance/tree/main/diffusers) supports **Mochi**, **HunyuanVideo**,**CogVideoX**,**SVD** and **LTX-Video** as of now
   
   To run the test script, refer to the `test.py` file in each folder. Below is an example using Mochi:
   
   ```python
   # test.py
   import torch
   from pipeline_stg_mochi import MochiSTGPipeline
   from diffusers.utils import export_to_video
   import os
   
   # Load the pipeline
   pipe = MochiSTGPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)
   
   pipe.enable_vae_tiling()
   pipe = pipe.to("cuda")
   
   #--------Option--------#
   prompt = "A slow-motion capture of a beautiful woman in a flowing dress spinning in a field of sunflowers, with petals swirling around her, realistic style."
   stg_mode = "STG-R" 
   stg_applied_layers_idx = [35]
   stg_scale = 0.8 # 0.0 for CFG (default)
   do_rescaling = True # False (default)
   #----------------------#
   
   # Generate video frames
   frames = pipe(
       prompt, 
       num_frames=84,
       stg_mode=stg_mode,
       stg_applied_layers_idx=stg_applied_layers_idx,
       stg_scale=stg_scale,
       do_rescaling=do_rescaling
   ).frames[0]
   ...
   ```
   For details on memory efficiency, inference acceleration, and more, refer to the original pages below:
   - [Mochi](https://huggingface.co/genmo/mochi-1-preview)
   - [CogVideoX](https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox)
   - [HunyuanVideo](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video)
   - [StableVideoDiffusion](https://huggingface.co/docs/diffusers/en/using-diffusers/svd)

## 🛠️Todos
- Implement STG on diffusers
- Update STG with Open-Sora, SVD

## 🙏Acknowledgements
This project is built upon the following works:
- [Mochi](https://github.com/genmoai/mochi?tab=readme-ov-file)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [LTX-Video](https://github.com/Lightricks/LTX-Video)
- [diffusers](https://github.com/huggingface/diffusers)

