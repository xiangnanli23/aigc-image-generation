""" see all schedulers: https://huggingface.co/docs/diffusers/en/api/schedulers/overview """

import os
import json
from typing import Union

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, 
    UniPCMultistepScheduler,
    EDMDPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler, 
)

def load_json(json_file: Union[str, os.PathLike]) -> dict:
    """ load json file into dict """
    with open(json_file, "r", encoding="utf-8") as json_file:
        dict = json.load(json_file)
    return dict


SCHEDULER_MAPPING = {
    "DDPM": DDPMScheduler,
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler,
    "DPM++ 2S a": DPMSolverSinglestepScheduler,
    "DPM++ 2S a Karras": DPMSolverSinglestepScheduler,
    "DPM++ SDE": DPMSolverSinglestepScheduler,
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 Karras": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
    "DPM2 a Karras": KDPM2AncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
    "LMS Karras": LMSDiscreteScheduler,
    "UNiPC": UniPCMultistepScheduler,
    "EDMDPM": EDMDPMSolverMultistepScheduler,

}
SCHEDULER_NAMES = list(SCHEDULER_MAPPING.keys())


def get_scheduler_class(scheduler_name: str) -> list:
    assert scheduler_name in SCHEDULER_NAMES, f"Unsupported sampler name: {scheduler_name}, available sampler names are {SCHEDULER_NAMES}"
    scheduler_cls = SCHEDULER_MAPPING[scheduler_name]
    return scheduler_cls


def build_scheduler(
    scheduler_name: str, 
    config: Union[dict, str],
    **kwargs,
    ):
    """ 
    Args:
        scheduler_name (`str`)
        config (`dict` or `str`):
            config class or config file path
    Kwargs:
        timestep_spacing (`str`)
    """
    # get
    timestep_spacing = kwargs.get("timestep_spacing", "leading")
    
    # get scheduler class
    scheduler_cls = get_scheduler_class(scheduler_name)

    if isinstance(config, str):
        config = load_json(config)
    if isinstance(config, dict):
        config = config
    
    # scheduler_kwargs
    scheduler_kwargs = {}
    scheduler_kwargs.update({"timestep_spacing": timestep_spacing})
    if "Karras" in scheduler_name:
        scheduler_kwargs.update({"use_karras_sigmas": True})
    if "SDE" in scheduler_name:
        scheduler_kwargs.update({"algorithm_type": "sde-dpmsolver++"})
    
    # build scheduler
    print(f"scheduler: {scheduler_name}, timestep_spacing: {timestep_spacing}")
    scheduler = scheduler_cls.from_config(
        config,
        **scheduler_kwargs,
    )
    return scheduler


def update_scheduler(
    pipe, 
    scheduler_name: str,
    **kwargs,
    ):
    """ 
    update scheduler, get scheduler config from pipe 
    Args:
        pipe: diffusers pipeline
        scheduler_name (`str`)
    Kwargs
        timestep_spacing (`str`)
    """
    scheduler_cls = get_scheduler_class(scheduler_name)
    if scheduler_cls in pipe.scheduler.compatibles:            
        scheduler = build_scheduler(scheduler_name, pipe.scheduler.config, **kwargs)
        pipe.scheduler = scheduler
    return pipe


