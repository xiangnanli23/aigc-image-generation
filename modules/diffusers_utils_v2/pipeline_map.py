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
    StableDiffusionAdapterPipeline,
    StableDiffusionXLAdapterPipeline,
)


#####################################################################################################
# sd15
SD15_PIPELINES = [
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,  # 'sdi2i'
    StableDiffusionInpaintPipeline,
]
SD15_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SD15_PIPELINES]
SD15_PIPELINE_MAP   = dict(zip(SD15_PIPELINE_NAMES, SD15_PIPELINES))

# sd15 cn
SD15_CN_PIPELINES = [
    StableDiffusionControlNetPipeline, 
    StableDiffusionControlNetImg2ImgPipeline, 
    StableDiffusionControlNetInpaintPipeline,
]
SD15_CN_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SD15_CN_PIPELINES]
SD15_CN_PIPELINE_MAP   = dict(zip(SD15_CN_PIPELINE_NAMES, SD15_CN_PIPELINES))

# sd adapter
SD15_ADAPTER_PIPELINES = [
    StableDiffusionAdapterPipeline,
]
SD15_ADAPTER_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SD15_ADAPTER_PIPELINES]
SD15_ADAPTER_PIPELINE_MAP   = dict(zip(SD15_ADAPTER_PIPELINE_NAMES, SD15_ADAPTER_PIPELINES))

# sdxl
SDXL_PIPELINES = [
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLInpaintPipeline,
]
SDXL_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SDXL_PIPELINES]
SDXL_PIPELINE_MAP   = dict(zip(SDXL_PIPELINE_NAMES, SDXL_PIPELINES))

# sdxl cn
SDXL_CN_PIPELINES = [
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
]
SDXL_CN_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SDXL_CN_PIPELINES]
SDXL_CN_PIPELINE_MAP   = dict(zip(SDXL_CN_PIPELINE_NAMES, SDXL_CN_PIPELINES))

# sdxl adpater
SDXL_ADAPTER_PIPELINES = [
    StableDiffusionXLAdapterPipeline,
]
SDXL_ADAPTER_PIPELINE_NAMES = [pipeline_class.__name__ for pipeline_class in SDXL_ADAPTER_PIPELINES]
SDXL_ADAPTER_PIPELINE_MAP = dict(zip(SDXL_ADAPTER_PIPELINE_NAMES, SDXL_ADAPTER_PIPELINES))


PIPELINE_MAP = {
    **SD15_PIPELINE_MAP, **SD15_CN_PIPELINE_MAP, 
    **SDXL_PIPELINE_MAP, **SDXL_CN_PIPELINE_MAP, 
    **SD15_ADAPTER_PIPELINE_MAP, **SDXL_ADAPTER_PIPELINE_MAP,
}


















