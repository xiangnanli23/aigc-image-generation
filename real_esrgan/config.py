
class Config:

    realesrgan_ckpt_dir = "/data6/lixiangnan/cv/sr/real_esrgan_ckpts"
    realesrgan_models = {
        'RealESRGAN_x4plus': f"{realesrgan_ckpt_dir}/RealESRGAN_x4plus.pth",
        'RealESRNet_x4plus': f"{realesrgan_ckpt_dir}/RealESRNet_x4plus.pth",
        'RealESRGAN_x4plus_anime_6B': f"{realesrgan_ckpt_dir}/RealESRGAN_x4plus_anime_6B.pth",
        'RealESRGAN_x2plus': f"{realesrgan_ckpt_dir}/RealESRGAN_x2plus.pth"
    }