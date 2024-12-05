# Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling

## Paper
- Arxiv: [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664)

## Project Page
- [STG Project Page](https://junhahyung.github.io/STGuidance)

## Todos
- Update README
- Update STG with Open-Sora, SVD, and LTX-Video

## Requirements
Please refer to the [original GitHub repository](https://github.com/genmoai/mochi?tab=readme-ov-file) for the requirements and setup.

## Acknowledgements
The code for the Mochi model is based on [Mochi](https://github.com/genmoai/mochi?tab=readme-ov-file).

## Start Guide
1. **Mochi**
   - For installation and requirements, refer to the [official repository](https://github.com/genmoai/mochi).
     
   - Update `config.py` with your desired settings and simply run:
     ```bash
     python ./demos/cli.py
     ```

2. **LTX-Video**
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
