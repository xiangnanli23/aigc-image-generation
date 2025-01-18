import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from typing import Union, List
from PIL import Image
from loguru import logger
from diffusers.models.embeddings import (
    IPAdapterFaceIDPlusImageProjection,
)

from handler.handler_sd_pipeline import StableDiffusionPipelineHandler
from utils_im_gen import (
    get_face_embeds,
    get_clip_embeds,
)

from real_esrgan.handler import RealESRGAN_Handler_V2

from utils.logger_utils import build_logger
from utils.flask_utils import convert_image
from utils.image_utils import resize_images
from utils.torch_utils import build_generator, clean_cache
from utils.oss_utils   import upload_image_urls



######################################################################
class RealESRGanModule:
    real_esrgan_handler = RealESRGAN_Handler_V2()

    def inference_real_esrgan(
        self, 
        images: List[Image.Image], 
        sr_model_name: str, 
        sr_model_path: str = None,
        sr_scale: float = 2.0,
        ) -> List[Image.Image]:
        
        # build model
        self.real_esrgan_handler.build_model(sr_model_name, sr_model_path)
        
        # inference loop
        unsampled_ims = []
        for image in images:
            unsampled_im = self.real_esrgan_handler.inference(image, outscale=sr_scale)
            unsampled_ims.append(unsampled_im)
        return unsampled_ims

class LoggingModule:
    enable_logging = False

    oss_env = "soulapp"

    def _build_logger(self, log_dir: str) -> None:
        build_logger(log_dir)
        self.enable_logging = True
    
    def info_build_args(self, pipeline_class, ckpt_path, scheduler_name, **kwargs) -> None:

        controlnet_path = kwargs.get("controlnet_path", None)
        
        lora_ckpt_paths = kwargs.get("lora_ckpt_paths", None)
        adapter_weights = kwargs.get("adapter_weights", None)
        
        ip_adapter_dir = kwargs.get("ip_adapter_dir", None)
        ip_adapter_ckpt_name_or_path = kwargs.get("ip_adapter_ckpt_name_or_path", None)
        ip_adapter_scale = kwargs.get("ip_adapter_scale", None)
        
        # logging
        build_args = {
            "pipeline_class": pipeline_class,
            "ckpt_path":      ckpt_path,
            "scheduler_name": scheduler_name,
        }
        
        if controlnet_path:
            build_args.update({
                "controlnet_path": controlnet_path
            })

        if lora_ckpt_paths:
            build_args.update({
                "lora_ckpt_paths": lora_ckpt_paths,
                "adapter_weights": adapter_weights,
            })

        if ip_adapter_dir and ip_adapter_ckpt_name_or_path:
            build_args.update({
                "ip_adapter_dir": ip_adapter_dir,
                "ip_adapter_ckpt_name_or_path": ip_adapter_ckpt_name_or_path,
                "ip_adapter_scale": ip_adapter_scale,
            })

        logger.info(f"build_args: {build_args}")

    def info_infer_args(self, **kwargs) -> None:

        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", None)

        width  = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        
        num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
        seed = kwargs.get("seed", -1)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        num_inference_steps = kwargs.get("num_inference_steps", 25)
        
        image      = kwargs.get('image', None)
        mask_image = kwargs.get('mask_image', None)
        strength   = kwargs.get('strength', 0.5)
        
        control_image                 = kwargs.get('control_image', None)
        controlnet_conditioning_scale = kwargs.get('controlnet_conditioning_scale', 0.5)
        control_guidance_start        = kwargs.get('control_guidance_start', 0.0)
        control_guidance_end          = kwargs.get('control_guidance_end', 1.0)

        ip_adapter_image = kwargs.get('ip_adapter_image', None)
        ip_adapter_scale = kwargs.get('ip_adapter_scale', 0.5)
        
        if image:
            image = upload_image_urls([image], oss_env=self.oss_env)
        if mask_image:
            mask_image = upload_image_urls([mask_image], oss_env=self.oss_env)
        if control_image:
            control_image = upload_image_urls([control_image], oss_env=self.oss_env)
        if ip_adapter_image:
            ip_adapter_image = upload_image_urls([ip_adapter_image], oss_env=self.oss_env)

        sr_model_name = kwargs.get('sr_model_name', None)
        sr_scale      = kwargs.get('sr_scale', 2)

        resize_w      = kwargs.get('resize_w', None)
        resize_h      = kwargs.get('resize_h', None)
        resize_method = kwargs.get('resize_method', None)

        infer_args = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            
            'width':  width,
            'height': height,
            
            'num_images_per_prompt': num_images_per_prompt,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            
            'image': image,
            'mask_image': mask_image,
            'strength': strength,
            
            'control_image': control_image,
            'controlnet_conditioning_scale': controlnet_conditioning_scale,
            'control_guidance_start': control_guidance_start,
            'control_guidance_end': control_guidance_end,  
            
            'ip_adapter_image': ip_adapter_image,
            'ip_adapter_scale': ip_adapter_scale,

            'sr_model_name': sr_model_name,
            'sr_scale': sr_scale,

            'resize_w': resize_w,
            'resize_h': resize_h,
            'resize_method': resize_method,

        }

        logger.info(f"infer_args: {infer_args}")

    def info_im_urls(self, im_gen_urls: List[str]) -> None:
        logger.info(f"im_gen_urls: {im_gen_urls}")
        pass

######################################################################
class StableDiffusionInference:
    """ build inference kwargs """
    default_prompt = None
    default_negative_prompt = None

    def _init_infer_kwargs(self, **kwargs) -> dict:
        """
        Basic text-2-image parameters
        Args:
            prompt
            negative_prompt
            width
            height
            num_images_per_prompt
            seed
            guidance_scale
            num_inference_steps
        """
        # parse and default
        prompt = kwargs.get("prompt", self.default_prompt)
        negative_prompt = kwargs.get("negative_prompt", self.default_negative_prompt)

        width  = kwargs.pop("width", 512)
        height = kwargs.pop("height", 512)
        
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
        seed = kwargs.pop("seed", -1)
        guidance_scale = kwargs.pop("guidance_scale", 7.5)
        num_inference_steps = kwargs.pop("num_inference_steps", 25)

        # build infer_kwargs
        infer_kwargs = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            
            'width':  width,
            'height': height,
            
            'num_images_per_prompt': num_images_per_prompt,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
        }

        generator, seed = build_generator(seed, return_seed=True)
        infer_kwargs.update({'generator': generator, 'seed': seed})

        return infer_kwargs, kwargs

    def _process_prompts(self, infer_kwargs: dict) -> dict:
        """ 
        use compel to process prompts 
        """
        if hasattr(self.pipeline_handler, 'compel'):
            prompt = infer_kwargs.pop("prompt", self.default_prompt)
            negative_prompt = infer_kwargs.pop("negative_prompt", self.default_negative_prompt)
            # SD15
            if self.pipeline_handler.model_type == "SD15":
                conditioning = self.pipeline_handler.compel(prompt)
                nega_conditioning = self.pipeline_handler.compel(negative_prompt)
                if conditioning is not None:
                    infer_kwargs.update({
                        'prompt_embeds': conditioning,
                        'negative_prompt_embeds': nega_conditioning,
                    })
            # SDXL
            if self.pipeline_handler.model_type in ["SDXL", "Playground"]:
                conditioning, pooled = self.pipeline_handler.compel(prompt)
                nega_conditioning, nega_pooled = self.pipeline_handler.compel(negative_prompt)
                if conditioning is not None:
                    infer_kwargs.update({
                        'prompt_embeds': conditioning,
                        'pooled_prompt_embeds': pooled,
                        'negative_prompt_embeds': nega_conditioning,
                        'negative_pooled_prompt_embeds': nega_pooled,
                    })
        return infer_kwargs

    def _process_images(self,  infer_kwargs: dict, **kwargs) -> dict:
        """
        Args:
            width
            height
            image
            mask_image
            strength
            control_image
            controlnet_conditioning_scale
            control_guidance_start
            control_guidance_end
            ip_adapter_image
            ip_adapter_scale
        """
        # get
        num_images_per_prompt = infer_kwargs.get('num_images_per_prompt', 1)

        width  = infer_kwargs.get('width', 512)
        height = infer_kwargs.get('height', 512)
        
        image      = kwargs.pop('image', None)
        mask_image = kwargs.pop('mask_image', None)
        strength   = kwargs.pop('strength', 0.5)
        
        control_image                 = kwargs.pop('control_image', None)
        controlnet_conditioning_scale = kwargs.pop('controlnet_conditioning_scale', 0.5)
        control_guidance_start        = kwargs.pop('control_guidance_start', 0.0)
        control_guidance_end          = kwargs.pop('control_guidance_end', 1.0)

        ip_adapter_ckpt_path = kwargs.pop('ip_adapter_ckpt_path', None)
        ip_adapter_image = kwargs.pop('ip_adapter_image', None)
        ip_adapter_scale = kwargs.pop('ip_adapter_scale', 0.5)

        image            = convert_image(image, return_type='image')
        mask_image       = convert_image(mask_image, return_type='image')
        control_image    = convert_image(control_image, return_type='image')
        ip_adapter_image = convert_image(ip_adapter_image, return_type='image')

        # image-2-image
        if ("Img2Img" in self.pipeline_handler.pipe_config['pipeline_name'] and
            isinstance(image, Image.Image)
        ):    
            image = image.resize((width, height))
            infer_kwargs.update({
                'image': image,
                'strength': strength,
            })

        # inpainting
        if ("Inpaint" in self.pipeline_handler.pipe_config['pipeline_name'] and
            isinstance(image, Image.Image) and 
            isinstance(mask_image, Image.Image)
        ):
            image      = image.resize((width, height))
            mask_image = mask_image.resize((width, height), Image.NEAREST)
            infer_kwargs.update({
                'image': image,
                'mask_image': mask_image,
                'strength': strength,
            })

        # controlnet
        if ("ControlNet" in self.pipeline_handler.pipe_config['pipeline_name'] and 
            isinstance(control_image, Image.Image)
        ):
            control_image = control_image.resize((width, height))
            infer_kwargs.update({
                'control_image': control_image,
                'controlnet_conditioning_scale': controlnet_conditioning_scale,
                'control_guidance_start': control_guidance_start,
                'control_guidance_end': control_guidance_end,  
            })
            if not (
                "Img2Img" in self.pipeline_handler.pipe_config['pipeline_name'] or
                "Inpaint" in self.pipeline_handler.pipe_config['pipeline_name']
            ):
                infer_kwargs.update({
                    'image': control_image,
                })

        # ip-adapter
        if isinstance(ip_adapter_image, Image.Image):
            # if load image encoder
            # ip_adapter_image = ip_adapter_image.resize((width, height))
            # infer_kwargs.update({
            #     'ip_adapter_image': ip_adapter_image,
            #     'ip_adapter_scale': ip_adapter_scale,
            # })

            clip_embeds = None
            if 'faceid' in ip_adapter_ckpt_path:
                face_embeds = get_face_embeds(ip_adapter_image)
                clip_embeds = get_clip_embeds(ip_adapter_image)
                # face_embeds = repeat_tensor(face_embeds, num_images_per_prompt)
                # clip_embeds = repeat_tensor(clip_embeds, num_images_per_prompt)

                infer_kwargs.update({
                    "ip_adapter_image_embeds": [face_embeds],
                })
                if clip_embeds is not None:
                    self.pipeline_handler.pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
                    if isinstance(self.pipeline_handler.pipe.unet.encoder_hid_proj.image_projection_layers[0], IPAdapterFaceIDPlusImageProjection):
                        self.pipeline_handler.pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut       = True # True if Plus v2
                        self.pipeline_handler.pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut_scale = 1.0
            else:
                clip_embeds = get_clip_embeds(ip_adapter_image)
                # clip_embeds = repeat_tensor(clip_embeds, num_images_per_prompt)
                infer_kwargs.update({
                    "ip_adapter_image_embeds": [clip_embeds],
                })

        return infer_kwargs, kwargs

    def build_infer_kwargs(self, **kwargs) -> dict:
        """
        Args:
            prompt
            negative_prompt
            width
            height
            num_images_per_prompt
            seed
            guidance_scale
            num_inference_steps

            width
            height
            image (`str` or `Image.Image`): URL or Base64 or Image.Image
            mask_image
            strength
            control_image
            controlnet_conditioning_scale
            control_guidance_start
            control_guidance_end
            ip_adapter_image
            ip_adapter_scale

        """
        infer_kwargs, rest_kwargs = self._init_infer_kwargs(**kwargs)
        infer_kwargs = self._process_prompts(infer_kwargs)
        infer_kwargs, rest_kwargs = self._process_images(infer_kwargs, **rest_kwargs)
        return infer_kwargs, rest_kwargs


class StableDiffusionImageGeneration(
    StableDiffusionInference,
    RealESRGanModule,
    LoggingModule,
    ):
    """ 
        Image Generation with Stable Diffusion
        Features
         - use pipeline_handler to manage pipeline
         - support logging
         - support super-resolution with RealESRGAN
         - support hard-resize
    """
    pipeline_handler = StableDiffusionPipelineHandler()
    
    def __init__(
        self, 
        log_dir: str = None,
        ) -> None:
        if log_dir:
            self._build_logger(log_dir)
            
    def build_pipeline(self, **kwargs) -> None:
        """
        Args:
            pipeline_class (`str` or `DiffusionPipeline`):
                pipeline class name
            ckpt_path (`str`):
                ckpt path
            scheduler_name (`str`):
                scheduler name
        Kwargs:
            dtype (`str`)
            device (`str`)

            enable_xformers_memory_efficient_attention (`bool`)
            enable_model_cpu_offload (`bool`)
            enable_vae_slicing (`bool`)

            num_in_channels (`int`)

            build_compel (`bool`)

            controlnet_path (`str`)

            lora_ckpt_paths (List[`str`])
            adapter_weights (List[`float`])
            
            ip_adapter_dir (`str`)
            ip_adapter_ckpt_name_or_path (`str`)
            ip_adapter_scale (`float`)
        """
        # parse kwargs
        pipeline_class = kwargs.pop('pipeline_class', "StableDiffusionPipeline")
        ckpt_path      = kwargs.pop('ckpt_path', None)
        scheduler_name = kwargs.pop('scheduler_name', "DDIM")

        # build pipeline
        self.pipeline_handler.build_pipeline(
            pipeline_class,
            ckpt_path,
            scheduler_name,
            **kwargs,
        )

        # logging
        if self.enable_logging:
            self.info_build_args(pipeline_class, ckpt_path, scheduler_name, **kwargs)
            print(f"##################################################################################################")

    def generate(self, **kwargs) -> Union[List[Image.Image], List[str]]:
        """
        Args:
            prompt
            negative_prompt
            width
            height
            num_images_per_prompt
            seed
            guidance_scale
            num_inference_steps

            image
            mask_image
            strength
            control_image
            controlnet_conditioning_scale
            control_guidance_start
            control_guidance_end
            
            ip_adapter_ckpt_path
            ip_adapter_image
            ip_adapter_scale
        
        Kwargs:
            sr_model_name
            sr_model_path
            sr_scale
            
            resize_w
            resize_h
            resize_method

            oss_env
            image_suffix
            upload_url
            return_url
        """
        # get
        sr_model_name = kwargs.get('sr_model_name', None)
        sr_model_path = kwargs.get('sr_model_path', None)
        sr_scale      = kwargs.get('sr_scale', 2)

        resize_w      = kwargs.get('resize_w', None)
        resize_h      = kwargs.get('resize_h', None)
        resize_method = kwargs.get('resize_method', "LANCZOS")

        oss_env       = kwargs.pop('oss_env', "soulapp")
        image_suffix  = kwargs.pop('image_suffix', 'png')
        upload_url    = kwargs.pop('upload_url', True)
        return_url    = kwargs.pop('return_url', True)
        
        clean_cache()
        
        # image generation
        infer_kwargs, rest_kwargs = self.build_infer_kwargs(**kwargs)
        im_gens: list = self.pipeline_handler.pipe(**infer_kwargs).images
        
        # upsample
        if sr_model_name is not None:
            im_gens = self.inference_real_esrgan(im_gens, sr_model_name, sr_model_path, sr_scale)

        # resize
        if resize_w:
            if isinstance(resize_w, float):
                resize_w = int(resize_w)
            if isinstance(resize_h, float):
                resize_h = int(resize_h)
            if (isinstance(resize_w, int) and isinstance(resize_h, int)):
                im_gens = resize_images(im_gens, resize_w, resize_h, resize_method)

        # logging
        if self.enable_logging:
            self.info_infer_args(**{**infer_kwargs, **rest_kwargs})
            print(f"##################################################################################################")

        # upload oss
        im_gen_urls = None
        if upload_url:
            if oss_env == 'playme':
                image_suffix = 'jpg'
            im_gen_urls = upload_image_urls(im_gens, image_suffix=image_suffix, oss_env=oss_env)

        clean_cache()
        
        # return
        if return_url:
            # logging
            if self.enable_logging:
                self.info_im_urls(im_gen_urls)
                print(f"##################################################################################################")
            return im_gens, im_gen_urls
        return im_gens

    def build_and_gen(self, **kwargs) -> Union[List[Image.Image], List[str]]:
        """
        Kwargs:
            pipeline_class (`str` or `DiffusionPipeline`):
                None 
            ckpt_path: (`str`):
                None
            scheduler_name (`str`):
                None

            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                None
            width (`int`):
                None
            height (`int`):
                None
            num_images_per_prompt (`int`):
                None
            seed (`int`):
                None
            guidance_scale (`float`):
                None
            num_inference_steps (`int`):
                None
            
            upload_url (`bool`): default True
            return_url (`bool`): default True
        
        """
        self.build_pipeline(**kwargs)
        im_gens = self.generate(**kwargs)
        return im_gens

