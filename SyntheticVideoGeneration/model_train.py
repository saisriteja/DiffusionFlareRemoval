################
# import stuff #
################

import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from safetensors import safe_open
import torch
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# napm needs to be imported before Animatediff
import napm

import sys
sys.path.append('AnimateDiff')

from AnimateDiff.animatediff.models.unet import UNet3DConditionModel
from AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline
from AnimateDiff.animatediff.utils.util import save_videos_grid
from AnimateDiff.animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from AnimateDiff.animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora



fpath_sd_model = "models/StableDiffusion/" # @param {type:"string"}
fpath_dreambooth_lora = "models/DreamBooth_LoRA/" # @param {type:"string"}
base_model = "Stable Diffusion v1.5" # @param ["Stable Diffusion v1.5", "Stable Diffusion v1.4"]
fpath_motion_prior = "models/Motion_Module/"



# if not Path(fpath_motion_prior).exists():
#     # !mkdir -p {fpath_motion_prior}
#     cmd = f"mkdir -p {fpath_motion_prior}"
#     os.system(cmd)
#     for mm_fname in ("mm_sd_v14.ckpt", "mm_sd_v15.ckpt"):
#         # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/{mm_fname} -d {fpath_motion_prior} -o {mm_fname}
#         cmd = f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/{mm_fname} -d {fpath_motion_prior} -o {mm_fname}"
#         os.system(cmd)

# if not Path(fpath_dreambooth_lora).exists():
#     # !mkdir -p {fpath_dreambooth_lora}
#     cmd = f"mkdir -p {fpath_dreambooth_lora}"
#     os.system(cmd)
#     # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/toonyou_beta3.safetensors -d {fpath_dreambooth_lora} -o toonyou_beta3.safetensors
#     cmd = f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/toonyou_beta3.safetensors -d {fpath_dreambooth_lora} -o toonyou_beta3.safetensors"
#     os.system(cmd)



train_config = OmegaConf.load("AnimateDiff/configs/training/v1/training.yaml")


mm_fname = "mm_sd_v15.ckpt"

lora_dreambooth_model_path = str(list(Path(fpath_dreambooth_lora).glob('*.safetensors'))[0])
motion_module_path = str(Path(fpath_motion_prior)/mm_fname)
#unet_additional_kwargs = {}


torch_dtype=torch.float16

tokenizer    = CLIPTokenizer.from_pretrained(fpath_sd_model, subfolder="tokenizer")#, torch_dtype=torch_dtype)
text_encoder = CLIPTextModel.from_pretrained(fpath_sd_model, subfolder="text_encoder" )#, torch_dtype=torch_dtype)
vae          = AutoencoderKL.from_pretrained(fpath_sd_model, subfolder="vae")#, torch_dtype=torch_dtype)
unet         = UNet3DConditionModel.from_pretrained_2d(fpath_sd_model,
                                                        subfolder="unet", 
                                                       unet_additional_kwargs=OmegaConf.to_container(train_config.get("unet_additional_kwargs", {})))


pipeline = AnimationPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=DDIMScheduler(**OmegaConf.to_container(train_config.noise_scheduler_kwargs)),
    ).to("cuda:1")


# probably wanna change this
func_args = SimpleNamespace

motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
if "global_step" in motion_module_state_dict:
  func_args.update({"global_step": motion_module_state_dict["global_step"]})
missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
assert len(unexpected) == 0


if lora_dreambooth_model_path != "":
    if lora_dreambooth_model_path.endswith(".ckpt"):
        state_dict = torch.load(lora_dreambooth_model_path)
        pipeline.unet.load_state_dict(state_dict)

    elif lora_dreambooth_model_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(lora_dreambooth_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        is_lora = all("lora" in k for k in state_dict.keys())
        if not is_lora:
            base_state_dict = state_dict
        else:
            base_state_dict = {}
            # TODO: deal with model_config.base
            with safe_open(model_config.base, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_state_dict[key] = f.get_tensor(key)

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
        pipeline.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        #pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)


        if is_lora:
            pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

pipeline.to("cuda")
# print(pipeline)





# @title Generate Animation

n_frames = 16 # @param {type:"integer"}
width = 448 # @param {type:"integer"}
height = 320 # @param {type:"integer"}

# TODO: maybe pl.seed_everything could improve determinism
seed = 10788741199826055526 # @param {type:"integer"}
steps = 2 # @param {type:"integer"}
guidance_scale = 7.5 # @param {type:"number"}
prompt = "apollo 13, 'houston we have a problem', tom hanks playing an astronaut" # @param {type:"string"}
negative_prompt = "stationary, motionless, boring, watermark, trademark, copyright, text, shutterstock" # @param {type:"string"}

outname = "sample.gif" # @param {type:"string"}
outpath = f"./{outname}"

from accelerate import PartialState
distributed_state = PartialState()
pipeline.to(distributed_state.device)

# print("Generating video...")

# while True:
#     pass


sample = pipeline(
    prompt=prompt,
    negative_prompt     = negative_prompt,
    num_inference_steps = steps,
    guidance_scale      = guidance_scale,
    width               = width,
    height              = height,
    video_length        = n_frames,
).videos


samples = torch.concat([sample])
save_videos_grid(samples, outpath , n_rows=1)

from IPython.display import Image

Image(outpath)