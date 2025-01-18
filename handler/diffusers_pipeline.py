import os
import time
import numpy as np
from PIL import Image
import torch
import torch.cuda

# sd15
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline, 
    StableDiffusionControlNetImg2ImgPipeline, 
    StableDiffusionControlNetInpaintPipeline,
)
# sdxl
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
# adpater
from diffusers import (
    ControlNetModel,
    T2IAdapter,
    StableDiffusionAdapterPipeline,
    StableDiffusionXLAdapterPipeline,
)
from diffusers.models import AutoencoderKL





################################################################################################
def build_generator(seed: int = -1, device=None, return_seed=False):
    if not isinstance(seed, int):  # like float
        seed = int(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if seed == -1:
        seed = np.random.randint(0, pow(2, 32))
    generator = torch.Generator(device).manual_seed(seed)
    if return_seed:
        return generator, seed
    else:
        return generator




# pipeline
# ################################################################################################
def build_pipeline(
    model_dir_or_path: str = None,
    model_type: str = 'sd15',
    pipe_type: str = 't2i',
    enable_vae_tiling: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
):
    """
    Args:
        model_type
            'sd15' or 'sdxl'
        pipe_type
            't2i' or 'i2i' or 'inpainting'
    """
    time_s = time.time()

    # decide pipeline class
    if model_type == 'sd15':
        if pipe_type == "t2i":
            pipeline_class = StableDiffusionPipeline
        if pipe_type == "i2i":
            pipeline_class = StableDiffusionImg2ImgPipeline
        if pipe_type == "inpainting":
            pipeline_class = StableDiffusionInpaintPipeline
        # original_config_file = '/data2/lixiangnan/work/aigc-all/config_files/sd_configs/cldm_v15.yaml'
    
    if model_type == 'sdxl':
        if pipe_type == "t2i":
            pipeline_class = StableDiffusionXLPipeline
        if pipe_type == "i2i":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        if pipe_type == "inpainting":
            pipeline_class = StableDiffusionXLInpaintPipeline
        # original_config_file = '/data2/lixiangnan/work/aigc-all/config_files/sd_configs/sd_xl_base.yaml'
    
    # build pipeline
    if os.path.isdir(model_dir_or_path):
        pipe = pipeline_class.from_pretrained(
            model_dir_or_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
    else:
        pipe = pipeline_class.from_single_file(
            model_dir_or_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
            # original_config_file=original_config_file,
        )
    
    pipe.to("cuda")
    
    if enable_vae_tiling:
        pipe.enable_vae_tiling()
    if enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    
    print("build pipeline time:", time.time() - time_s)
    return pipe

def build_cn_pipeline(
    pipe,
    model_type,
    pipe_type,
    controlnet_ckpt_dir_or_path,
):
    """    
    Args:
        model_type
            'sd15' or 'sdxl'
        pipe_type
            't2i' or 'i2i' or 'inpainting'
    """
    # build controlnet
    if os.path.isdir(controlnet_ckpt_dir_or_path):
        controlnet = ControlNetModel.from_pretrained(
            controlnet_ckpt_dir_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        controlnet = ControlNetModel.from_single_file(
            controlnet_ckpt_dir_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    controlnet.to('cuda')

    # build pipeline
    if model_type == 'sd15':
        if pipe_type == 't2i':
            pipeline_class = StableDiffusionControlNetPipeline
        if pipe_type == 'i2i':
            pipeline_class = StableDiffusionControlNetImg2ImgPipeline
        if pipe_type == 'inpainting':
            pipeline_class = StableDiffusionControlNetInpaintPipeline
        pipe_cn = pipeline_class(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            controlnet=controlnet,
        )
    
    if model_type == 'sdxl':
        if pipe_type == 't2i':
            pipeline_class = StableDiffusionXLControlNetPipeline
        if pipe_type == 'i2i':
            pipeline_class = StableDiffusionXLControlNetImg2ImgPipeline
        if pipe_type == 'inpainting':
            pipeline_class = StableDiffusionXLControlNetInpaintPipeline
        pipe_cn = pipeline_class(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            feature_extractor=None,
            controlnet=controlnet,
        )
    
    pipe_cn.to('cuda')
    pipe_cn.enable_xformers_memory_efficient_attention()
    return pipe_cn

def convert_pipeline(
    pipe, 
    model_type: str, 
    return_pipe_type: str,
):
    """
    return a new pipe instance
    Args:
        model_type
            'sd15' or 'sdxl'
        return_pipe_type
            't2i' or 'i2i' or 'inpainting'
    """

    if model_type == 'sd15':
        if return_pipe_type == "t2i":
            pipe_class = StableDiffusionPipeline
        if return_pipe_type == "i2i":
            pipe_class = StableDiffusionImg2ImgPipeline
        if return_pipe_type == "inpainting":
            pipe_class = StableDiffusionInpaintPipeline
    
        new_pipe = pipe_class(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            feature_extractor=pipe.feature_extractor,
            image_encoder=pipe.image_encoder,
            safety_checker=None,
            requires_safety_checker=False,
        )

    if model_type == 'sdxl':
        if return_pipe_type == "t2i":
            pipe_class = StableDiffusionXLPipeline
        if return_pipe_type == "i2i":
            pipe_class = StableDiffusionXLImg2ImgPipeline
        if return_pipe_type == "inpainting":
            pipe_class = StableDiffusionXLInpaintPipeline
    
        new_pipe = pipe_class(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            feature_extractor=pipe.feature_extractor,
            image_encoder=pipe.image_encoder,
        )

    new_pipe.to("cuda")
    new_pipe.enable_xformers_memory_efficient_attention()
    return new_pipe

def update_vae(pipe):
    vae_path = '/data6/lixiangnan/models/vae/sd-vae-ft-mse'
    vae_path = '/data6/lixiangnan/models/vae/sdxl-vae-fp16-fix'
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to('cuda')
    pipe.vae = vae
    pipe.to("cuda")
    pipe.to(dtype=torch.float16)
    return pipe

# scheduler
################################################################################################

################################################################################################
def load_lora(pipe, lora_ckpt_path: str, lora_scale: float = 0.5):
    """
    only load one lora
    """
    time_s = time.time()

    pipe.load_lora_weights(lora_ckpt_path)    
    pipe.set_adapters(adapter_weights=lora_scale)
    pipe.to('cuda')

    print("load lora time:", time.time() - time_s)
    return pipe
    

################################################################################################
def inference(
    pipe, 
    im_size: tuple = (896, 1312), 
    seed: int = 666,
) -> str:
    
    generator = build_generator(seed)
    
    im_path = '/data2/lixiangnan/work/aigc-all/tmp/image/0d918190-8eed-4218-bf84-9d3d0525c09a.png'
    # image = Image.open(im_path).resize((521, 1024))

    infer_kwargs = {
        "prompt": "a dog",
        "negative_prompt": "ugly",
        "width":  im_size[0],
        "height": im_size[1],
        "num_images_per_prompt": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "generator": generator,
        # "image": image,
        "strength": 0.5,
    }
    image = pipe(**infer_kwargs).images[0]
    return image


def main():
    """ build + inference """
    # build pipe
    # model_dir_or_path = '/data6/lixiangnan/models/ckpt/anything_v50.safetensors'
    # model_type = 'sd15'
    model_dir_or_path = '/data6/lixiangnan/models/ckpt_sdxl/sd_xl_base_1.0.safetensors'
    model_type = 'sdxl'

    pipe_type = 't2i'
    
    pipe = build_pipeline(
        model_dir_or_path=model_dir_or_path,
        model_type=model_type,
        pipe_type=pipe_type,
        enable_xformers_memory_efficient_attention=True,
    )

    image = inference(pipe)
    # image.save('/data2/lixiangnan/output.jpg')
    print('done')







if __name__ == '__main__':
    main()
    pass















