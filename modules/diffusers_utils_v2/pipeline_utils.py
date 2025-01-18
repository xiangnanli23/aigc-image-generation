import os
import sys

from typing import Union, List
import yaml
import torch
import json

from transformers import (
    CLIPTokenizer,
    CLIPTextConfig, 
    CLIPTextModel,
)
from diffusers.models import ControlNetModel


from .loaders_v27.single_file_utils import create_diffusers_controlnet_model_from_ldm
from .sdxl_utils import (
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
)
from .ckpt_utils import load_checkpoint



SD_CONFIG_DIR_PATH           = os.path.join(os.path.dirname(__file__), 'config_files', 'sd_configs')
TEXT_ENCODER_CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), 'config_files', 'text_encoder_configs')
CLIP_TOKENIZER_DIR_PATH      = os.path.join(os.path.dirname(__file__), "clip_tokenizers")
SD_CONFIG_FILES = {
    "v1": f"{SD_CONFIG_DIR_PATH}/v1-inference.yaml",
    "v2": f"{SD_CONFIG_DIR_PATH}/v2-inference-v.yaml",
    "xl": f"{SD_CONFIG_DIR_PATH}/sd_xl_base.yaml",
    "xl_refiner": f"{SD_CONFIG_DIR_PATH}/sd_xl_refiner.yaml",
    "upscale": f"{SD_CONFIG_DIR_PATH}/x4-upscaling.yaml",
    "controlnet": f"{SD_CONFIG_DIR_PATH}/cldm_v15.yaml",
}
SD15_TEXT_ENCODER_CONFIG_FILE = f"{TEXT_ENCODER_CONFIG_DIR_PATH}/sd15_text_encoder_config.json"

# SD15
SD15_TOKENIZER_PATH = os.path.join(CLIP_TOKENIZER_DIR_PATH, "sd15_tokenizer")

# SDXL
TOKENIZER_1_PATH = os.path.join(CLIP_TOKENIZER_DIR_PATH, "clip-vit-large-patch14")
TOKENIZER_2_PATH = os.path.join(CLIP_TOKENIZER_DIR_PATH, "CLIP-ViT-bigG-14-laion2B-39B-b160k")
TEXT_ENCODER_1_CONFIG_PATH = os.path.join(TOKENIZER_1_PATH, "config.json")
TEXT_ENCODER_2_CONFIG_PATH = os.path.join(TOKENIZER_2_PATH, "config.json")


def load_json(json_file: Union[str, os.PathLike]) -> dict:
    """ load json file into dict """
    assert os.path.splitext(json_file)[1].lower() == '.json', f"{json_file} is not a json file!"
    with open(json_file, "r", encoding="utf-8") as json_file:
        dict = json.load(json_file)
    return dict






def convert_ldm_clip_checkpoint_v1(checkpoint):
  keys = list(checkpoint.keys())
  text_model_dict = {}
  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]
  return text_model_dict


def fetch_original_config(original_config_file: str) -> dict:
    with open(original_config_file, "r") as fp:
        original_config_file = fp.read()
    original_config = yaml.safe_load(original_config_file)
    return original_config




class SDComponentsBuilder:
    """ stable diffusion 15 """

    @staticmethod
    def build_tokenizer():
        tokenizer = CLIPTokenizer.from_pretrained(SD15_TOKENIZER_PATH)
        return tokenizer
    @staticmethod
    def build_text_encoder(checkpoint):
        config_dict = load_json(SD15_TEXT_ENCODER_CONFIG_FILE)
        text_encoder_config = CLIPTextConfig(**config_dict)
        text_encoder = CLIPTextModel(text_encoder_config)
        converted_vae_checkpoint = convert_ldm_clip_checkpoint_v1(checkpoint)
        text_encoder.load_state_dict(converted_vae_checkpoint, strict=False)
        return text_encoder



class SDXLComponentsBuilder:

    @staticmethod
    def build_tokenizer():
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_1_PATH, 
            local_files_only=True,
        )
        return tokenizer

    @staticmethod
    def build_text_encoder(checkpoint):
        text_encoder = convert_ldm_clip_checkpoint(
            checkpoint,
            config_path=TEXT_ENCODER_1_CONFIG_PATH,
        )
        return text_encoder

    @staticmethod
    def build_tokenizer_2():
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            TOKENIZER_2_PATH, 
            pad_token="!", 
            local_files_only=True,
        )
        return tokenizer_2

    @staticmethod
    def build_text_encoder_2(checkpoint, is_refiner=False):
        prefix = "conditioner.embedders.0.model." if is_refiner else "conditioner.embedders.1.model."
        text_encoder_2 = convert_open_clip_checkpoint(
            checkpoint,
            config_path=TEXT_ENCODER_2_CONFIG_PATH,
            prefix=prefix,
            has_projection=True,
        )
        return text_encoder_2



class ControlnetBuilder:
    @staticmethod
    def build_cn(
        controlnet_path: Union[str, List[str]],
        torch_dtype=torch.float16, 
        local_files_only=True,
        ):
        if isinstance(controlnet_path, list):
            controlnet_path = controlnet_path
        else:
            controlnet_path = [controlnet_path]

        controlnets = []
        for cn_path in controlnet_path:
            if os.path.isdir(cn_path):
                controlnet = ControlNetModel.from_pretrained(
                    cn_path, 
                    torch_dtype=torch_dtype, 
                    local_files_only=local_files_only)
            if isinstance(cn_path, str) and cn_path.endswith('.pth'):
                controlnet = ControlNetModel.from_single_file(
                    cn_path, 
                    torch_dtype=torch_dtype, 
                    local_files_only=local_files_only)
            controlnets.append(controlnet)
        
        if len(controlnets) == 1:
            return controlnets[0]
        return controlnets

    @staticmethod
    def build_cn_v2(
        ckpt_path: str,
        **kwargs,
        ) -> ControlNetModel:

        dtype  = kwargs.get('dtype', torch.float16)
        device = kwargs.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        
        upcast_attention = kwargs.get("upcast_attention", False)
        image_size = kwargs.get("image_size", None)
        
        original_config_file = SD_CONFIG_FILES['controlnet']
        original_config = fetch_original_config(original_config_file)

        checkpoint = load_checkpoint(ckpt_path, dtype=dtype)

        component = create_diffusers_controlnet_model_from_ldm(
            ControlNetModel, 
            original_config, 
            checkpoint, 
            upcast_attention=upcast_attention, 
            image_size=image_size,
        )
        controlnet = component["controlnet"]
        controlnet.to(dtype)
        controlnet.to(device)
        return controlnet



if __name__ == '__main__':
    pass


