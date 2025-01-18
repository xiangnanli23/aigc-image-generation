import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def inference(path, outscale, save_path):

    netscale = 4
    model_path = "/data2/liuyibo/AIGC/sd-git-project/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth"
    dni_weight = None
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    tile = 0
    tile_pad = 10
    pre_pad = 0
    gpu_id = None
    fp32 = False
    
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # if len(img.shape) == 3 and img.shape[2] == 4:
    #     img_mode = 'RGBA'
    # else:
    #     img_mode = None
    
    output, _ = upsampler.enhance(img, outscale=outscale)
    
    print()


if __name__ == '__main__':
    
    inference(
        path="/data2/liuyibo/AIGC/sd-git-project/Real-ESRGAN/inputs/00003.png",
        outscale=2,
        save_path=""
    )
    