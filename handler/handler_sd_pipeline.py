import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from typing import Union, List
import time

from diffusers import (
    DiffusionPipeline,
)

from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)

from configs.config import SD_CONFIG_FILES
# from configs.path_config import IP_ADAPTER_DIR

from modules.diffusers_utils_v2.loaders_v27.single_file import (
    build_sub_model_components, 
    set_additional_components,
)
from modules.diffusers_utils_v2.pipeline_utils import (
    fetch_original_config,
    SDComponentsBuilder,
    SDXLComponentsBuilder,
    ControlnetBuilder,
)
from modules.diffusers_utils_v2.scheduler_utils import SchedulerBuilder
from modules.diffusers_utils_v2.ckpt_utils import load_checkpoint
from modules.diffusers_utils_v2.pipeline_map import (
    PIPELINE_MAP,
    SD15_PIPELINES,
    SD15_CN_PIPELINES,
    SD15_ADAPTER_PIPELINES,
    SDXL_PIPELINES,
    SDXL_CN_PIPELINES,
    SDXL_ADAPTER_PIPELINES,
)

from utils.signature_utils import get_signature_keys
from utils.torch_utils import clean_cache



############################################################################
class PipelineHandlerUtils:

    def clean_cache(self):
        clean_cache()

class PipelineHandlerModules:
    compel = None

    def _build_compel(self):
        from compel import Compel, ReturnedEmbeddingsType

        # sd15
        if self.model_type == "SD15":
            self.compel = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
            )
        # sdxl
        if self.model_type == "SDXL" or self.model_type == "Playground":
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )

############################################################################
class LoraConfig:
    lora_config = {
        "lora_ckpt_names": [],  # list
        "adapter_names": [],
        "adapter_weights": [],
    }
    def clear_lora_config(self):
        self.lora_config = {
            "lora_ckpt_names": [],  # list
            "adapter_names": [],
            "adapter_weights": [],
        }

class PipelineHandlerLoraLoader(LoraConfig):

    def _load_loras(
        self, 
        lora_ckpt_paths: List[str],
        adapter_weights: List[float],
        ) -> None:
        """
        Args:
            lora_ckpt_paths (List[str])
            adapter_weights (List[float])
        """
        lora_ckpt_names = [os.path.basename(lora_ckpt_path) for lora_ckpt_path in lora_ckpt_paths]
        print(f"Loras: {lora_ckpt_names}, lora_scales: {adapter_weights}")
        
        self.remove_loras()
        
        # load loras
        adapter_names = []
        for lora_ckpt_path in lora_ckpt_paths:
            adapter_name = f"lora_{len(adapter_names)}"
            self.pipe.load_lora_weights(lora_ckpt_path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
        self.pipe.set_adapters(
            adapter_names=adapter_names, 
            adapter_weights=adapter_weights,
        )
        
        # update config
        self.lora_config['lora_ckpt_names'] = lora_ckpt_names
        self.lora_config['adapter_names']   = adapter_names
        self.lora_config['adapter_weights'] = adapter_weights
    
    def load_loras(
        self, 
        lora_ckpt_paths: List[str],
        adapter_weights: List[float],
        ) -> None:
        """
        Args:
            lora_ckpt_paths (List[str])
            adapter_weights (List[float])
        """
        self.enable_loras()

        lora_ckpt_names = [os.path.basename(lora_ckpt_path) for lora_ckpt_path in lora_ckpt_paths]

        if (lora_ckpt_names != self.lora_config['lora_ckpt_names'] or 
            adapter_weights != self.lora_config['adapter_weights']
        ):
            self._load_loras(lora_ckpt_paths, adapter_weights)

    def remove_loras(self):
        """ remove loaded loras """
        if len(self.lora_config['adapter_names']) != 0:
            self.pipe.delete_adapters(self.lora_config['adapter_names'])
            self.lora_config['adapter_names'] = []

    def disable_loras(self):
        self.pipe.disable_lora()

    def enable_loras(self): 
        self.pipe.enable_lora()

############################################################################
class IpAdapterConfig:
    ip_adapter_config = {
        "ip_adapter_load_status": False,
        "ip_adapter_ckpt_name": None,
        "ip_adapter_scale": None,
    }

    def clear_ip_adapter_config(self):
        self.ip_adapter_config = {
            "ip_adapter_load_status": False,
            "ip_adapter_ckpt_name": None,
            "ip_adapter_scale": None,
        }

class PipelineHandlerIpAdapterLoader(IpAdapterConfig):
    # ip_adapter_dir = IP_ADAPTER_DIR

    def _load_ip_adapter(
        self,
        ip_adapter_ckpt_path: str = None,
        ip_adapter_scale: float = 1.0,
        image_encoder_folder=None,
        ) -> None:
        """
        Load IP-Adapter (without loading iamge encoder)
        """
        
        if self.ip_adapter_config['ip_adapter_load_status']:
            self.pipe.unload_ip_adapter()
        
        ip_adapter_dir_path  = os.path.dirname(ip_adapter_ckpt_path)
        ip_adapter_ckpt_name = os.path.basename(ip_adapter_ckpt_path)
        
        print(f"IP-Adapter: {ip_adapter_ckpt_name}, ip_adapter_scale: {ip_adapter_scale}")

        # load ip adapter
        self.pipe.load_ip_adapter(
            ip_adapter_dir_path,
            subfolder=None, 
            weight_name=ip_adapter_ckpt_name,
            image_encoder_folder=image_encoder_folder,
        )
        # set ip adapter scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        
        # # set ip-adapter-faceid lora
        # if 'faceid_0' in self.pipe.get_active_adapters():
        #     self.pipe.set_adapters(
        #         adapter_names=[ip_adapter_ckpt_name],
        #         adapter_weights=[ip_adapter_scale],
        #     )

        # update config
        self.ip_adapter_config['ip_adapter_load_status'] = True
        self.ip_adapter_config['ip_adapter_ckpt_name']   = ip_adapter_ckpt_name
        self.ip_adapter_config['ip_adapter_scale']       = ip_adapter_scale

    def load_ip_adapter(
        self,
        ip_adapter_ckpt_path: str = None,
        ip_adapter_scale: float = 1.0,
        ) -> None:
        """
        Load IP-Adapter (without loading iamge encoder)
        """
        ip_adapter_ckpt_name = os.path.basename(ip_adapter_ckpt_path)
        if (ip_adapter_ckpt_name != self.ip_adapter_config['ip_adapter_ckpt_name'] or 
            ip_adapter_scale != self.ip_adapter_config['ip_adapter_scale']
        ):
            self._load_ip_adapter(ip_adapter_ckpt_path, ip_adapter_scale)
            
    def unload_ip_adapter(self):
        if self.ip_adapter_config['ip_adapter_load_status']:
            self.pipe.unload_ip_adapter()
            if 'faceid_0' in self.pipe.get_active_adapters():  # for ip-adapter-faceid which will load a lora
                self.pipe.unload_lora_weights()
        self.clear_ip_adapter_config()

    def load_ip_adapters(self, ip_adapter_ckpt_paths: Union[List[str], str], image_encoder_folder: Union[List[str], str] = None):
        """ load multiple ip-adapters without loading image encoder """
        weight_names = [os.path.basename(ip_adapter_ckpt_path) for ip_adapter_ckpt_path in ip_adapter_ckpt_paths]
        dir_paths    = [os.path.dirname(ip_adapter_ckpt_path) for ip_adapter_ckpt_path in ip_adapter_ckpt_paths]

        self.pipe.load_ip_adapter(
            dir_paths,
            subfolder=None, 
            weight_name=weight_names,
            image_encoder_folder=image_encoder_folder,
        )


############################################################################
class PipelineConfig:
    pipe_config = {
        "pipeline_name": None,
        "ckpt_name": None,
        "scheduler_name": None,
        "controlnet_ckpt_name": None,
    }
    ckpt_config = {
        'ckpt': None,
    }
    scheduler_config = {
        "timestep_spacing": None,
    }

class StableDiffusionPipelineHandler(
    PipelineConfig,
    PipelineHandlerModules,
    PipelineHandlerLoraLoader,
    PipelineHandlerIpAdapterLoader,
    ):
    """
    Manage the build, update of pipeline
    """

    pipe = None
    model_type = None  # "SD15", "SDXL", "Playground"

    def _is_rebuild_required(
        self,
        pipeline_class: Union[str, DiffusionPipeline],
        ckpt_path: str = None,
        scheduler_name: str = None,
        **kwargs,
        ) -> List[bool]:
        """
        whether to rebuild pipeline, controlnet, scheduler, etc.
        When to update pipeline:
            1. pipeline_class is changed or,
            2. ckpt_path is changed
        When to update controlnet:
            1. ControlNet Pipeline and,
            2. controlnet path is changed
        """
        update_pipeline   = False
        update_ckpt       = False
        update_controlnet = False
        update_scheduler  = False
        
        # update pipeline
        if isinstance(pipeline_class, str):
            pipeline_class = PIPELINE_MAP[pipeline_class]
        pipeline_name = pipeline_class.__name__
        if self.pipe_config["pipeline_name"] != pipeline_name:
            update_pipeline = True

        # update ckpt
        ckpt_name = os.path.basename(ckpt_path)
        if self.pipe_config["ckpt_name"] != ckpt_name:
            update_ckpt = True

        # update scheduler
        timestep_spacing = kwargs.get("timestep_spacing", None)
        if (self.pipe_config["scheduler_name"] != scheduler_name or 
            self.scheduler_config['timestep_spacing'] != timestep_spacing
        ):
            update_scheduler = True

        # update controlnet
        controlnet_path = kwargs.get("controlnet_path", None)
        controlnet_ckpt_name = os.path.basename(controlnet_path) if isinstance(controlnet_path, str) else None
        if ("ControlNet" in pipeline_name and 
            controlnet_path is not None and 
            self.pipe_config["controlnet_ckpt_name"] != controlnet_ckpt_name
        ):
            update_controlnet = True

        return update_pipeline, update_ckpt, update_scheduler, update_controlnet

    def _is_load_required(
        self,
        lora_ckpt_paths,
        adapter_weights,
        ip_adapter_ckpt_path,
        ip_adapter_scale,
        ):
        load_loras = False
        load_ip_adapter = False

        if (lora_ckpt_paths is not None and 
            len(lora_ckpt_paths) > 0 and
            len(lora_ckpt_paths) == len(adapter_weights)
        ):
            load_loras = True
        
        if (ip_adapter_ckpt_path is not None and 
            ip_adapter_scale is not None
        ):
            load_ip_adapter = True

        return load_loras, load_ip_adapter

    def _remove_check(
        self,
        lora_ckpt_paths,
        ip_adapter_ckpt_path,
        ):
        """
        if input are None Type, then remove it
        """
        remove_loras = False
        remove_ip_adapter = False
        if lora_ckpt_paths is None:
            remove_loras = True
        if ip_adapter_ckpt_path is None:
            remove_ip_adapter = True
        return remove_loras, remove_ip_adapter

    def _load_checkpoint(
        self,
        ckpt_path,
        **kwargs,
        ) -> None:
        # get
        dtype  = kwargs.get("dtype", torch.float16)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # load ckpt
        checkpoint = load_checkpoint(checkpoint_path=ckpt_path, device=device, dtype=dtype)
        self.ckpt_config.update({"ckpt": checkpoint})

    def _build_pipeline(
        self,
        pipeline_class: Union[str, DiffusionPipeline],
        ckpt_path: str = None,
        scheduler_name: str = None,
        **kwargs,
        ) -> None:
        """
        Build Pipeline
        Args:
            pipeline_class (`str` or `DiffusionPipeline`)
            ckpt_path (`str`)
            scheduler_name (`str`)
        Kwargs:
            dtype (`str`)
            device (`str`)

            enable_xformers_memory_efficient_attention (`bool`)
            enable_model_cpu_offload (`bool`)
            enable_vae_slicing (`bool`)

            num_in_channels (`int`)

            controlnet (`ControlNetModel`)
            controlnet_path (`str`)

            requires_safety_checker (`bool`)
        """

        if isinstance(pipeline_class, str):
            pipeline_class = PIPELINE_MAP[pipeline_class]
        class_name = pipeline_class.__name__
        ckpt_name   = os.path.basename(ckpt_path)
        print(f"pipeline: {class_name} ...")
        print(f"ckpt_name: {ckpt_name}")

        # get
        dtype  = kwargs.get("dtype", torch.float16)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        enable_xformers_memory_efficient_attention = kwargs.get("enable_xformers_memory_efficient_attention", True)
        enable_model_cpu_offload                   = kwargs.get('enable_model_cpu_offload', True)
        enable_vae_slicing                         = kwargs.get("enable_vae_slicing", True)
        num_in_channels     = kwargs.get("num_in_channels", 4)
        controlnet          = kwargs.get("controlnet", None)
        controlnet_path     = kwargs.get("controlnet_path", None)
        image_size          = kwargs.get("image_size", None)
        load_safety_checker = kwargs.get("load_safety_checker", False)

        # update
        kwargs.update({"requires_safety_checker": False})
        kwargs.update({"num_in_channels": num_in_channels})

        # load ckpt
        checkpoint = load_checkpoint(checkpoint_path=ckpt_path, device=device, dtype=dtype)

        # build controlnet
        if isinstance(controlnet_path, str):
            controlnet_ckpt_name = os.path.basename(controlnet_path)
            print(f"build controlnet: {controlnet_ckpt_name}")
            controlnet = ControlnetBuilder.build_cn_v2(controlnet_path, dtype=dtype, device=device)
        else:
            controlnet_ckpt_name = None
        kwargs["controlnet"] = controlnet

        # start building pipeline
        expected_modules, optional_modules = get_signature_keys(pipeline_class)
        print(f"pipeline expected modules: {expected_modules}")
        print(f"pipeline optional modules: {optional_modules}")
        passed_class_obj   = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_modules if k in kwargs}

        if pipeline_class in SD15_PIPELINES + SD15_CN_PIPELINES + SD15_ADAPTER_PIPELINES:
            model_type = "SD15"
            original_config_file = SD_CONFIG_FILES["v1"]
            tokenizer    = SDComponentsBuilder.build_tokenizer()
            text_encoder = SDComponentsBuilder.build_text_encoder(checkpoint)
            passed_class_obj.update({
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
            })

        if pipeline_class in SDXL_PIPELINES + SDXL_CN_PIPELINES + SDXL_ADAPTER_PIPELINES:
            model_type = "SDXL"
            original_config_file = SD_CONFIG_FILES["xl"]
            tokenizer      = SDXLComponentsBuilder.build_tokenizer()
            tokenizer_2    = SDXLComponentsBuilder.build_tokenizer_2()
            text_encoder   = SDXLComponentsBuilder.build_text_encoder(checkpoint)
            text_encoder_2 = SDXLComponentsBuilder.build_text_encoder_2(checkpoint)
            passed_class_obj.update({
                "tokenizer": tokenizer,
                "text_encoder": text_encoder,
                "tokenizer_2": tokenizer_2,
                "text_encoder_2": text_encoder_2,
            })

        # build config
        original_config = fetch_original_config(original_config_file)

        # for playground
        if "playground" in ckpt_path:
            model_type = "Playground"

        # start building
        init_kwargs = {}
        for name in expected_modules:
            if name in passed_class_obj:
                init_kwargs[name] = passed_class_obj[name]
            else:
                components = build_sub_model_components(
                    init_kwargs,
                    class_name,
                    name,
                    original_config,
                    checkpoint,
                    model_type=model_type,
                    image_size=image_size,
                    load_safety_checker=load_safety_checker,
                    local_files_only=True,
                    **kwargs,
                )
                if not components:
                    continue
                init_kwargs.update(components)

        additional_components = set_additional_components(
            class_name,
            original_config,
            model_type=model_type,
        )
        if additional_components:
            init_kwargs.update(additional_components)

        init_kwargs.update(passed_pipe_kwargs)

        # init pipe
        pipe: DiffusionPipeline = pipeline_class(**init_kwargs)
        pipe.to(device=device)
        pipe.to(dtype=dtype)

        if enable_xformers_memory_efficient_attention and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            print(f"enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            print(f"enable vae slicing")
            pipe.enable_vae_slicing()
        
        # update pipeline config
        self.pipe = pipe
        self.model_type = model_type
        self.pipe_config.update({
            "pipeline_name": class_name,
            "ckpt_name": os.path.basename(ckpt_path),
            "scheduler_name": scheduler_name,
            "controlnet_ckpt_name": controlnet_ckpt_name,
        })

    def _build_controlnet(
        self,
        controlnet_path: str,
        **kwargs,
        ):
        """
        Assuming it is already controlnet pipeline, update controlnet module
        Args:
            controlnet_path (`str`)
        Kwargs:
            dtype (`str`)
            device (`str`)
        """
        # get
        dtype  = kwargs.get("dtype", torch.float16)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        controlnet_ckpt_name = os.path.basename(controlnet_path)

        # build or update controlnet
        print(f"build controlnet: {controlnet_ckpt_name}")
        controlnet = ControlnetBuilder.build_cn_v2(
            controlnet_path,
            dtype=dtype,
            device=device,
        )
        self.pipe.controlnet = controlnet
        
        # update config
        self.pipe_config["controlnet_ckpt_name"] = controlnet_ckpt_name

    def _build_scheduler(
        self,
        scheduler_name: str,
        **kwargs,
        ):
        """
        Args:
            scheduler_name (`str`)
        Kwargs:
            timestep_spacing (`str`)
        """
        # get
        timestep_spacing = kwargs.get("timestep_spacing", None)
        
        # update scheduler
        self.pipe = SchedulerBuilder.update_scheduler(
            self.pipe,
            scheduler_name,
            **kwargs,
        )
        
        # update config
        self.pipe_config["scheduler_name"]        = scheduler_name
        self.scheduler_config['timestep_spacing'] = timestep_spacing

    def _convert_pipeline(
        self, 
        pipeline_class: Union[str, DiffusionPipeline],
        **kwargs,
        ) -> None:
        """
        use diffusers official api to convert pipeline class
        """
        # get
        dtype  = kwargs.get("dtype", torch.float16)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        enable_xformers_memory_efficient_attention = kwargs.get("enable_xformers_memory_efficient_attention", True)
        enable_model_cpu_offload                   = kwargs.get('enable_model_cpu_offload', True)
        enable_vae_slicing                         = kwargs.get("enable_vae_slicing", True)
        
        if isinstance(pipeline_class, str):
            pipeline_class = PIPELINE_MAP[pipeline_class]
        pipeline_name = pipeline_class.__name__
        
        if "Img2Img" in pipeline_name:
            pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        elif "Inpaint" in pipeline_name:
            pipe = AutoPipelineForInpainting.from_pipe(self.pipe)
        else:
            pipe = AutoPipelineForText2Image.from_pipe(self.pipe)

        pipe.to(device=device)
        pipe.to(dtype=dtype)
        if enable_xformers_memory_efficient_attention and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            print(f"enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            print(f"enable vae slicing")
            pipe.enable_vae_slicing()

        self.pipe = pipe
        self.pipe_config.update({"pipeline_name": pipeline_name})

    def build_pipeline(
        self,
        pipeline_class: Union[str, DiffusionPipeline],
        ckpt_path: str = None,
        scheduler_name: str = None,
        **kwargs,
        ) -> None:
        """ 
        build / update pipeline and scheduler 
        Args:
            pipeline_class (`str` or `DiffusionPipeline`):
                pipeline class name
            ckpt_path (`str`):
                ckpt path
            scheduler_name (`str`):
                scheduler name
        Kwargs:
            controlnet_path (`str`)
            dtype (`str`)
            device (`str`)

            enable_xformers_memory_efficient_attention (`bool`)
            enable_model_cpu_offload (`bool`)
            enable_vae_slicing (`bool`)

            num_in_channels (`int`)

            build_compel (`bool`)

            lora_ckpt_paths (List[`str`])
            adapter_weights (List[`float`])

            ip_adapter_ckpt_path (`str`)
            ip_adapter_scale (`float`)

        """
        clean_cache()
        
        # get
        controlnet_path = kwargs.get('controlnet_path', None)
        build_compel    = kwargs.get('build_compel', True)
        
        lora_ckpt_paths = kwargs.get('lora_ckpt_paths', None)
        adapter_weights = kwargs.get('adapter_weights', None)
        
        ip_adapter_ckpt_path = kwargs.get('ip_adapter_ckpt_path', None)
        ip_adapter_scale     = kwargs.get('ip_adapter_scale', 1.0)

        # update check
        update_pipeline, update_ckpt, update_scheduler, update_controlnet = self._is_rebuild_required(
            pipeline_class, 
            ckpt_path, 
            scheduler_name, 
            **kwargs
        )
        print('update_pipeline: ', update_pipeline)
        print('update_ckpt: ', update_ckpt)
        print('update_scheduler: ', update_scheduler)
        print('update_controlnet: ', update_controlnet)
        
        # load check
        load_loras, load_ip_adapter = self._is_load_required(
            lora_ckpt_paths,
            adapter_weights,
            ip_adapter_ckpt_path,
            ip_adapter_scale,
        )

        print('load_loras: ', load_loras)
        print('load_ip_adapter: ', load_ip_adapter)

        # remove check
        remove_loras, remove_ip_adapter = self._remove_check(
            lora_ckpt_paths,
            ip_adapter_ckpt_path,
        )

        print('remove_loras: ', remove_loras)
        print('remove_ip_adapter: ', remove_ip_adapter)

        # update
        if update_ckpt:
            self._build_pipeline(pipeline_class, ckpt_path, scheduler_name, **kwargs)
        elif update_pipeline:
            self._convert_pipeline(pipeline_class)

        if update_scheduler:
            self._build_scheduler(scheduler_name, **kwargs)
        
        if update_controlnet and (not update_pipeline):
            self._build_controlnet(controlnet_path, **kwargs)

        # load loras
        if load_loras:
            self.load_loras(lora_ckpt_paths, adapter_weights)

        # load ip-adapter
        if load_ip_adapter:
            self.load_ip_adapter(ip_adapter_ckpt_path, ip_adapter_scale)

        # remove loras
        if remove_loras:
            self.remove_loras()
        
        # remove ip adapter
        if remove_ip_adapter:
            self.unload_ip_adapter()
        
        # build compel
        if build_compel:
            self._build_compel()

        print("pipe active adapters: ", self.pipe.get_active_adapters())
        
