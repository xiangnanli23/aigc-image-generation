import os


class Config:
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)



# handler config
#################################################################################
SD_CONFIG_DIR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modules/diffusers_utils_v2/config_files', 'sd_configs')
SD_CONFIG_FILES = {
    "v1": f"{SD_CONFIG_DIR_PATH}/v1-inference.yaml",
    "v2": f"{SD_CONFIG_DIR_PATH}/v2-inference-v.yaml",
    "xl": f"{SD_CONFIG_DIR_PATH}/sd_xl_base.yaml",
    "xl_refiner": f"{SD_CONFIG_DIR_PATH}/sd_xl_refiner.yaml",
    "upscale": f"{SD_CONFIG_DIR_PATH}/x4-upscaling.yaml",
    "controlnet": f"{SD_CONFIG_DIR_PATH}/cldm_v15.yaml",
}

# ui config
#################################################################################
MODEL_TYPES = ['SD15', 'SDXL']

IMAGE_RATIOS = {
    "SD15": (512, 512),   # (W, H)
    "SDXL": (1024, 1024),
}

DEFAULT_SCHEDULER_NAME = "Euler a"
DEFAULT_NEGATIVE_PROMPT = "nsfw, (low quality, normal quality, worst quality, jpeg artifacts), cropped, monochrome, lowres, low saturation, ((watermark)), (white letters), skin spots, acnes, skin blemishes, age spot, mutated hands, mutated fingers, deformed, bad anatomy, disfigured, poorly drawn face, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, out of focus, long neck, long body, extra fingers, fewer fingers, (multi nipples), bad hands, signature, username, bad feet, blurry, bad body"

NONE = "None"


BASE_CHECKPOINT_DIRS = {
    "SD15": "/data6/lixiangnan/models/ckpt",
    "SDXL": "/data6/lixiangnan/models/ckpt_sdxl",
}

LORA_CHECKPOINT_DIRS = {
    "SD15": "/data6/lixiangnan/models/lora",
    "SDXL": "/data6/lixiangnan/models/lora_sdxl",
}

IP_ADAPTER_CHECKPOINT_DIRS = {
    "SD15": "/data6/lixiangnan/models/ip_adapter/sd15",
    "SDXL": "/data6/lixiangnan/models/ip_adapter/sdxl",
}

DEFAULT_BASE_CHECKPOINT = {
    "SD15": "sd15.ckpt",
    "SDXL": "sd_xl_base_1.0.safetensors",
}

DEFAULT_LORA_CHECKPOINT = {
    "SD15": "None",
    "SDXL": "None",
}

DEFAULT_IP_ADAPTER_CHECKPOINT = {
    "SD15": "None",
    "SDXL": "None",
}

OSS_ENVS = ["soulapp", "mysoulmate", "playme"]


DEFAULT_USER = [
    ('123', '123'),
    ('456', '456'),
]


# image generation demo config
#################################################################################
IMAGE_GENERATION_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_data")

os.makedirs(IMAGE_GENERATION_DEMO_DIR, exist_ok=True)
IMAGE_GENERATION_DEMO_UPLOAD_DIR = f'{IMAGE_GENERATION_DEMO_DIR}/uploads'
os.makedirs(IMAGE_GENERATION_DEMO_UPLOAD_DIR, exist_ok=True)
IMAGE_GENERATION_DEMO_RESULT_DIR = f'{IMAGE_GENERATION_DEMO_DIR}/results'
os.makedirs(IMAGE_GENERATION_DEMO_RESULT_DIR, exist_ok=True)
