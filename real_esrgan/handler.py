import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

from real_esrgan.config import Config

from modules.real_esrgan.realesrgan import RealESRGANer
from utils.torch_utils import clean_cache


REALESRGAN_CKPT_DIR = Config.realesrgan_ckpt_dir
REALESRGAN_MODELS = Config.realesrgan_models



class RealesrganHandler():
    
    def __init__(self, model_path) -> None:
        self.model = self._build_model(model_path)
        pass

    def _build_model(self, model_path):
        highres_model = RRDBNet(num_in_ch=3, 
                                num_out_ch=3, 
                                num_feat=64, 
                                num_block=6, 
                                num_grow_ch=32, 
                                scale=4)
        
        netscale = 4
        dni_weight = None
        highres_tile = 0
        tile_pad = 10
        pre_pad = 0
        gpu_id = None
        fp32 = False
        
        model = RealESRGANer(netscale, 
                                model_path, 
                                dni_weight, 
                                highres_model, 
                                highres_tile, 
                                tile_pad, 
                                pre_pad, 
                                not fp32, 
                                gpu_id)
        return model

    
    def inference(self, image: Image.Image, outscale) -> np.array:
        image = np.array(image)
        image_upsampled, _ = self.model.enhance(image, outscale=outscale)
        return image_upsampled


class RealESRGAN_Config:
    model_name = None

class RealESRGAN_Handler_V2(RealESRGAN_Config):
    upsampler = None

    def _build_model(
        self, 
        model_name: str,
        model_path: str = None,
        ) -> None:
        clean_cache()

        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            
        if model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            
        if model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            
        if model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            
        if model_path is None:
            model_path = REALESRGAN_MODELS[model_name]

        dni_weight = None
        tile = 0
        tile_pad = 10
        pre_pad = 0
        gpu_id = None
        fp32 = False

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id,
        )
        
        # update config
        self.model_name = model_name

    def build_model(self, model_name: str, model_path: str = None) -> None:
        if model_name != self.model_name:
            self._build_model(model_name, model_path)

    def inference(self, image: Image.Image, outscale: float = 2) -> Image.Image:
        image = np.array(image)
        image_upsampled, _ = self.upsampler.enhance(image, outscale=outscale)
        image_upsampled    = Image.fromarray(image_upsampled)
        return image_upsampled




def example():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    model_path = "/data2/liuyibo/AIGC/sd-git-project/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth"
    model_path = '/data6/lixiangnan/cv/sr/real_esrgan/RealESRGAN_x4plus_anime_6B.pth'
    real_esrgan_handler = RealesrganHandler(model_path)
    im_path = '/data2/lixiangnan/work/aigc-all/test/test_images/female-cute_1_512X768.png'
    outscale = 2.3
    image = Image.open(im_path)
    image_upsampled = real_esrgan_handler.inference(image, outscale)
    image_upsampled = Image.fromarray(image_upsampled)
    
    image_upsampled.save('real_esrgan_upsampled_23.png')
    pass


if __name__ == "__main__":
    # example()
    pass