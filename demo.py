"""
    CUDA_VISIBLE_DEVICES="" SERVER_PORT= python 

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

SERVER_PORT = int(os.environ.get("SERVER_PORT", 2324))

import torch
from typing import List, Union
import gradio as gr
from PIL import Image
from loguru import logger
from tqdm import tqdm

from utils.gradio_utils import GradioUIBase
from modules.diffusers_utils_v2.scheduler_utils import SCHEDULER_NAMES

from handler.sd_image_generation import StableDiffusionImageGeneration
from utils_im_gen import (
    login,
    update_default,
    get_pipeline_t2i,
    get_pipeline_i2i,
    UI_Components,
)

from configs.config import (
    Config,
    MODEL_TYPES,
    DEFAULT_SCHEDULER_NAME,
    DEFAULT_NEGATIVE_PROMPT,
    BASE_CHECKPOINT_DIRS,
    LORA_CHECKPOINT_DIRS,
    OSS_ENVS,
)



# init image generation
#########################################################################################################
sd_im_generator = StableDiffusionImageGeneration(Config.log_dir)


#########################################################################################################
def inference_demo(
    model_type,
    base_ckpt_name,
    lora_ckpt_name, lora_weight,
    scheduler_name,  
    prompt, negative_prompt,
    height, width,
    num_images_per_prompt, seed,
    guidance_scale, num_inference_steps,

    enable_sr, sr_model_name, sr_scale,
    enable_hard_resize, hard_resize_h, hard_resize_w,
    
    oss_env,
    
    image = None, strength = None,

    return_url: bool = False,
):

    im_gens, im_gen_urls = inference(
        model_type,
        base_ckpt_name,
        lora_ckpt_name, lora_weight,
        scheduler_name,  
        prompt, negative_prompt,
        height, width,
        num_images_per_prompt, seed,
        guidance_scale, num_inference_steps,
        enable_sr, sr_model_name, sr_scale,
        enable_hard_resize, hard_resize_h, hard_resize_w,
        oss_env,
        image, strength,
    )
    if return_url:
        return im_gen_urls
    else:
        return im_gens


#########################################################################################################
def inference(
    model_type,
    base_ckpt_name,
    lora_ckpt_name, lora_weight,
    scheduler_name,  
    prompt, negative_prompt,
    height, width,
    num_images_per_prompt, seed,
    guidance_scale, num_inference_steps,
    
    enable_sr, sr_model_name, sr_scale,
    enable_hard_resize, hard_resize_h, hard_resize_w,
    
    oss_env,
    
    image, strength,
    ) -> List[Image.Image]:
    """
    parse inputs -> generate images -> return images
    """
    
    # decide pipeline
    pipeline_class = get_pipeline_t2i(model_type)
    if image:
        pipeline_class = get_pipeline_i2i(model_type)
    ckpt_path = os.path.join(BASE_CHECKPOINT_DIRS[model_type], base_ckpt_name)

    # infer kwargs
    infer_kwargs = {
        "pipeline_class": pipeline_class,
        "ckpt_path": ckpt_path,   
        "scheduler_name": scheduler_name,

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        
        "height": height,
        "width": width,
        
        "num_images_per_prompt": num_images_per_prompt,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,

        "oss_env": oss_env,
        "upload_url": True,
        "return_url": True,
    }

    # image 2 image
    if image:
        infer_kwargs.update({
            "image": image,
            "strength": strength,
        })

    # lora
    if lora_ckpt_name != "None":
        lora_ckpt_path = os.path.join(LORA_CHECKPOINT_DIRS[model_type], lora_ckpt_name)
        infer_kwargs.update({
            "lora_ckpt_paths": [lora_ckpt_path],
            "adapter_weights": [lora_weight],
        })

    # upscale - real-esrgan
    if enable_sr:
        infer_kwargs.update({
            "sr_model_name": sr_model_name,
            "sr_scale": sr_scale,
        })
    
    # hard resize
    if enable_hard_resize:
        infer_kwargs.update({
            "resize_w": hard_resize_w,
            "resize_h": hard_resize_h,
        })

    # image generation
    im_gens, im_gen_urls = sd_im_generator.build_and_gen(**infer_kwargs)

    return im_gens, im_gen_urls


#########################################################################################################
class UI(UI_Components):

    def _ui_text2image(self):
        """
        """
        # ------------------------------------------------------------------------------------------ #
        gr.Markdown("""
            <h2> Before Generation </h2>
        """, visible=False)
        with gr.Accordion(label="OSS ENV", open=False, visible=False):
            with gr.Row():
                oss_env = gr.Radio(choices=OSS_ENVS, label="OSS", value=OSS_ENVS[0])
        
        # ------------------------------------------------------------------------------------------ #
        gr.Markdown("""
            <h2> Text Prompts </h2>
        """)
        prompt          = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=4)
        
        # ------------------------------------------------------------------------------------------ #
        gr.Markdown("""
            <h2> Advanced Parameters & Generated Images </h2>
        """)
        with gr.Row():  # advanced parameters
            with gr.Column():
                with gr.Column():
                    # base model
                    with gr.Column():
                        model_type = gr.Radio(label="Model Type", choices=MODEL_TYPES)
                    with gr.Column():
                        base_ckpt_name = gr.Dropdown(label="Base Checkpoint")
                    
                    # lora
                    lora_ckpt_name, lora_weight = self._ui_lora()
                    
                    # ip-adapter
                    ip_adapter_ckpt_name, ip_adapter_scale, ip_adapter_image = self._ui_ip_adapter()
                    
                    with gr.Accordion(label="Advanced"):
                        with gr.Row():
                            scheduler_name = gr.Dropdown(label="scheduler", choices=SCHEDULER_NAMES, value=DEFAULT_SCHEDULER_NAME)
                        with gr.Row():
                            with gr.Column():
                                height = gr.Slider(label="height (分辨率 - 高)", minimum=256, maximum=2048, step=8, value=1024)
                                width  = gr.Slider(label="width (分辨率 - 宽)", minimum=256, maximum=2048, step=8, value=1024)
                                num_images_per_prompt = gr.Slider(label="num_gen_images (每个文本生成张数)", minimum=1, maximum=10, step=1, value=2)        
                            with gr.Column():
                                seed                = gr.Slider(label="seed (随机种子)", minimum=-1, maximum=2147483647, step=1, value=-1)
                                guidance_scale      = gr.Slider(label="guidance_scale (prompt权重系数)", minimum=0.0, maximum=20, value=7.5, step=0.5)
                                num_inference_steps = gr.Slider(label="num_inference_steps (迭代步数)", minimum=1, maximum=70, value=30, step=1)
                    
                ####### -------------------------------------------------------------------- #######
                # Upscaling - real-esrgan
                with gr.Row():
                    enable_sr, sr_model_name, sr_scale = self._ui_module_real_esrgan()
                
                # Postprocessing - hard resize
                with gr.Row():
                   enable_hard_resize, hard_resize_h, hard_resize_w = self._ui_module_hard_resize()

                ####### -------------------------------------------------------------------- #######
            
            # output - generated images
            # -------------------------------------------------------------------------------- #
            with gr.Tab("gen_images"):
                with gr.Column():
                    output_images = gr.Gallery(label='gen_images', format='png')
    
        # ------------------------------------------------------------------------------------------ #
        with gr.Column():
            run_button = gr.Button(value="Generation", variant='primary')
    
        # -------------------------------------------------------------------------------- #
        inputs  = [
                model_type,
                base_ckpt_name,
                lora_ckpt_name, lora_weight,
                scheduler_name,  
                prompt, negative_prompt,
                height, width,
                num_images_per_prompt, seed,
                guidance_scale, num_inference_steps,



                enable_sr, sr_model_name, sr_scale,
                enable_hard_resize, hard_resize_h, hard_resize_w,
                
                oss_env,
        ]
        outputs = [output_images]
        run_button.click(fn=inference_demo, inputs=inputs, outputs=outputs)

        # -------------------------------------------------------------------------------- #
        model_type.select(update_default, inputs=[model_type], outputs=[base_ckpt_name, lora_ckpt_name, ip_adapter_ckpt_name, width, height])


#########################################################################################################
class Demo(GradioUIBase, UI):

    def start(self, server_name, server_port, default_concurrency_limit):
        block = gr.Blocks().queue(default_concurrency_limit=default_concurrency_limit)
        with block:
            self._title("AIGC - Stable Diffusion - Image Generation")
            self._build_tab("text2image", self._ui_text2image)
            
            # log out 
            # logout_button = gr.Button("Log Out", link="/logout")

        block.launch(
            server_name=server_name, 
            server_port=server_port, 
            show_error=True,
            # auth=login,
            # auth_message="Enter username and password for login",
        )


#########################################################################################################
def main():
    server_name = '0.0.0.0'
    server_port = SERVER_PORT
    default_concurrency_limit = 1
    demo = Demo()
    demo.start(server_name, server_port, default_concurrency_limit)


#########################################################################################################
if __name__ == "__main__":
    main()
    pass
