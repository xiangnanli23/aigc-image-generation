import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from typing import Union
import gradio as gr
from loguru import logger
from PIL import Image

from configs.config import (
    Config,
    IMAGE_RATIOS,
    BASE_CHECKPOINT_DIRS,
    LORA_CHECKPOINT_DIRS,
    IP_ADAPTER_CHECKPOINT_DIRS,
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_LORA_CHECKPOINT,
    DEFAULT_IP_ADAPTER_CHECKPOINT,
    NONE,
    IMAGE_GENERATION_DEMO_DIR,
    DEFAULT_USER,
)

from real_esrgan.handler import REALESRGAN_MODELS

from modules.api.api_openai_custom import get_chatgpt_response as get_llm_response


from utils.http_utils import HTTP
from utils.torch_utils import list2tensor
from utils.oss_utils import upload_image_url
from utils.flask_utils import convert_image

########################################################################################################################
def login(username: str, password: str) -> bool:
    if (username, password) in DEFAULT_USER:
        return True
    else:
        return False


def get_listdir(dir_path: str) -> list:
    return os.listdir(dir_path)

def get_ckpt_names(model_type: str):
    """
    """
    base_ckpt_dir = BASE_CHECKPOINT_DIRS[model_type]
    lora_ckpt_dir = LORA_CHECKPOINT_DIRS[model_type]
    ip_adapter_ckpt_dir = IP_ADAPTER_CHECKPOINT_DIRS[model_type]
    base_ckpt_name = gr.Dropdown(label="Base Checkpoint", 
                                 choices=get_listdir(base_ckpt_dir), 
                                 value=DEFAULT_BASE_CHECKPOINT[model_type])
    lora_ckpt_name = gr.Dropdown(label="Lora Checkpoint", 
                                 choices=[NONE] + get_listdir(lora_ckpt_dir), 
                                 value=DEFAULT_LORA_CHECKPOINT[model_type])
    ip_adapter_ckpt_name = gr.Dropdown(label="IP-Adapter Checkpoint", 
                                 choices=[NONE] + get_listdir(ip_adapter_ckpt_dir), 
                                 value=DEFAULT_IP_ADAPTER_CHECKPOINT[model_type])
    return base_ckpt_name, lora_ckpt_name, ip_adapter_ckpt_name

def get_w_h(model_type: str):
    width, height = IMAGE_RATIOS[model_type] 
    width  = gr.Slider(label="width (分辨率 - 宽)", minimum=256, maximum=2048, step=8, value=width)
    height = gr.Slider(label="height (分辨率 - 高)", minimum=256, maximum=2048, step=8, value=height)
    return width, height



def update_default(model_type):
    """
    update below:
        base_ckpt_name
        lora_ckpt_name
        height & width
            SD15 - 512
            SDXL - 1024
    """
    base_ckpt_name, lora_ckpt_name, ip_adapter_ckpt_name = get_ckpt_names(model_type)
    width, height = get_w_h(model_type)
    return base_ckpt_name, lora_ckpt_name, ip_adapter_ckpt_name, width, height



def get_pipeline_t2i(model_type: str):
    pipeline = None
    if model_type == "SD15":
        pipeline = "StableDiffusionPipeline"
    if model_type == "SDXL":
        pipeline = "StableDiffusionXLPipeline"
    return pipeline

def get_pipeline_i2i(model_type: str):
    pipeline = None
    if model_type == "SD15":
        pipeline = "StableDiffusionImg2ImgPipeline"
    if model_type == "SDXL":
        pipeline = "StableDiffusionXLImg2ImgPipeline"
    return pipeline



########################################################################################################################
def get_avatar_image(
    image: str,  # image url
    avatar_model_type, 
    expand_factor_w, expand_factor_h, 
    avatar_enable_hard_resize, 
    avatar_hard_resize_width, avatar_hard_resize_height, 
    return_highest_confidence,
    oss_env,
) -> Union[str, None]:
    """ this is an api """

    url = Config.avatar_image_api_url
    
    data: dict = {
        "data_params": {
            "model_type": avatar_model_type,
            "image": image,
            "expand_factor_w": expand_factor_w,
            "expand_factor_h": expand_factor_h,
            "avatar_enable_hard_resize": avatar_enable_hard_resize,
            "avatar_hard_resize_width": avatar_hard_resize_width,
            "avatar_hard_resize_height": avatar_hard_resize_height,
            "return_highest_confidence": return_highest_confidence,
            "oss_env": oss_env,
        },
    }
    try:
        res = HTTP.post(url, data=data)
        avatar_urls = res['data']['avatar_urls']
        if len(avatar_urls) > 0:
            return avatar_urls[0]
        else:
            return None
    except:
        return None
    
########################################################################################################################
def get_face_embeds(image: Image.Image):
    """ for ip-adapter """
    data = {
        "data_params": {
            'image': convert_image(image, return_type='url'),
        },
    }
    try:
        res = HTTP.post(Config.face_embeds_url, data=data)
        return list2tensor(res['data']['embedding'])
    except:
        return torch.zeros(2, 1, 1, 512).to('cuda').to(torch.float16)


def get_clip_embeds(image: Image.Image, output_hidden_states=True, do_classifier_free_guidance=True):
    """ for ip-adapter """
    data = {
        "data_params": {
            'image': convert_image(image, return_type='url'),
            'output_hidden_states': output_hidden_states,
            'do_classifier_free_guidance': do_classifier_free_guidance,
        },
    }
    try:
        res = HTTP.post(Config.clip_embeds_url, data=data)
        return list2tensor(res['data']['embedding'])
    except:
        if output_hidden_states:
            return torch.zeros(2, 1, 257, 1280).to('cuda').to(torch.float16)
        else:
            return torch.zeros(2, 1, 1024).to('cuda').to(torch.float16)



########################################################################################################################

class UI_Introduction:
    """
    <h2> Recommended Image Ratio </h2>
    <ol>
        <li> 512 X 512 </li>
        <li> 512 X 768 </li>
    </ol>
    """
    def _ui_introduction(self):
        """ AIGC - Image Generation Demo """
        gr.Markdown(
            """
            <h1> Welcome to AIGC - Image Generation Demo! </h1>
            Author: Xiangnan Li - Witter

            <h2> Related Documents </h2>
            <ol>
                <li> <a href='https://soulapp.feishu.cn/wiki/AnKXw18vmiRuj2k9NMucgKatnIc?from=from_copylink'> introduction to image generation demo </a> </li>
            </ol>

            <h2> Related Demos </h2>
            <ol>
                <li> Avatar Image Generation - http://10.30.92.78:2602 </li>
            </ol>

            """
        )





UPLOAD_CKPT_TYPE = ['sd15_base', 'sdxl_base', 'sd15_lora', 'sdxl_lora']
UPLOAD_CKPT_SUFFIX = ['safetensors', 'ckpt', 'pt', 'pth']

UPLOAD_CKPT_DIR = None
UPLOAD_CKPT_DIR_MAP = {
    'sd15_base': BASE_CHECKPOINT_DIRS["SD15"], 
    'sdxl_base': BASE_CHECKPOINT_DIRS["SDXL"], 
    'sd15_lora': LORA_CHECKPOINT_DIRS["SD15"], 
    'sdxl_lora': LORA_CHECKPOINT_DIRS["SDXL"],
}
class UI_UploadCkpts:
    
    def _update_upload_dir_path(self, upload_ckpt_type: str):
        global UPLOAD_CKPT_DIR
        UPLOAD_CKPT_DIR = UPLOAD_CKPT_DIR_MAP[upload_ckpt_type]

    def _process_uploaded_file(self, files: list) -> str:
        global UPLOAD_CKPT_DIR
        if len(files) != 1:
            gr.Warning("Please just upload one file at a time!")
            upload_result = "Upload failed! Please just upload one file at a time!"
        elif not files[0].name.split('.')[-1] in UPLOAD_CKPT_SUFFIX:
            gr.Warning(f"uploading failed: {files[0].name.split('.')[-1]} is not supported. Supported data types are {UPLOAD_CKPT_SUFFIX}")            
            upload_result = f"uploading failed: {files[0].name.split('.')[-1]} is not supported. Supported data types are {UPLOAD_CKPT_SUFFIX}"
        else:
            gr.Info("Start uploading... Wait for a while and do not click the button again!!!")
            file_paths = [file.name for file in files]
            file_path = file_paths[0]
            logger.info(f'uploaded_file_path: {file_path}')
            file_name = os.path.basename(file_path)
            os.system(f"cp '{file_path}' '{UPLOAD_CKPT_DIR}/{file_name}'")
            gr.Info("Successfully upload a file!")
            logger.info(f'successfully upload_file: {file_name}')
            
            os.system(f"rm '{file_path}'")
            upload_result = "Successfully upload your ckpt!"
        return upload_result

    def _ui_upload_ckpts(self):
        """ """
        gr.Markdown(
            """
            <h2> What you should know before you upload: </h2> 
            <ol>
                <li> Uploading time depends on the size of the uploaded ckpt and the Internet.  </li>
                <li> A ckpt whose size is 1 G could take about 20 mins, during the uploading time do not close the browser!  </li>
            </ol>
            """
        )
        with gr.Column():    
            upload_ckpt_type = gr.Dropdown(label="Ckpt Type", choices=UPLOAD_CKPT_TYPE)
            upload_button = gr.UploadButton(label="Upload a File", file_types=["file"], file_count="multiple", variant='primary')
            upload_result = gr.Textbox(label='Uploading Result', type='text')

        upload_ckpt_type.change(self._update_upload_dir_path, inputs=[upload_ckpt_type])
        upload_button.upload(self._process_uploaded_file, upload_button, upload_result)



class UI_FileExplorer:

    def _run_file_explorer(self, file: list):
        return file

    def _ui_module_file_explorer(self):
        file_explorer = gr.FileExplorer(
            label='Results',
            root_dir=IMAGE_GENERATION_DEMO_DIR,
        )
        run_button_file_explorer = gr.Button("Get Chosen Files")
        choosen_files = gr.File(label='chosen_files')
        
        inputs  = [file_explorer]
        outputs = [choosen_files]
        run_button_file_explorer.click(fn=self._run_file_explorer, inputs=inputs, outputs=outputs)
        



class UI_Components:
    
    def _ui_lora(self):
        with gr.Accordion(label="Lora", open=False):
            with gr.Row():
                lora_ckpt_name = gr.Dropdown(label="Lora Checkpoint")
            with gr.Row():
                lora_weight = gr.Slider(label="lora weight", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
        return lora_ckpt_name, lora_weight

    def _ui_ip_adapter(self):
        with gr.Accordion(label="IP-Adapter & IP-Adapter-FaceID", open=False):
            with gr.Column():
                ip_adapter_ckpt_name = gr.Dropdown(label="IP-Adapter Checkpoint")
                ip_adapter_scale = gr.Slider(label="ip_adapter_scale", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
            with gr.Row():
                ip_adapter_image = gr.Image(label="ip_adapter_image", type='pil')

        return ip_adapter_ckpt_name, ip_adapter_scale, ip_adapter_image

    def _ui_prompt_refiner(self):
        with gr.Accordion(label="Prompt Refiner", open=False):
            with gr.Row():
                enable_refined_prompt = gr.Checkbox(label="enable_refined_prompt", value=False)
            with gr.Row():
                llm_instruction = gr.Textbox(label="LLM-Instruction", value=LLM_INSTRUCTION, lines=20)
                refined_prompt = gr.Textbox(label="Refined Prompt")
            with gr.Row():    
                refine_prompt_run_button = gr.Button(value="Generation", variant='primary')
        return enable_refined_prompt, llm_instruction, refined_prompt, refine_prompt_run_button


    def _ui_module_real_esrgan(self):
        with gr.Accordion(label="Upscaling - Real ESRGAN", open=False):
            with gr.Column():
                enable_sr = gr.Checkbox(label="enable_sr", value=False)
            with gr.Row():
                sr_model_name = gr.Dropdown(label="sr_model", choices=list(REALESRGAN_MODELS.keys()), value="RealESRGAN_x4plus_anime_6B")
                sr_scale = gr.Slider(label="sr_scale", minimum=1.0, maximum=4.0, value=2.0, step=0.1, scale=2)
        return enable_sr, sr_model_name, sr_scale


    def _ui_module_hard_resize(self):
        with gr.Accordion(label="Postprocessing - Hard Resize", open=False):
            with gr.Row():
                enable_hard_resize = gr.Checkbox(label="enable_hard_resize", value=False)
                hard_resize_height = gr.Slider(label="height", minimum=1, maximum=2048, step=1, value=1600)
                hard_resize_width  = gr.Slider(label="width", minimum=1, maximum=2048, step=1, value=900)
        return enable_hard_resize, hard_resize_height, hard_resize_width
    

