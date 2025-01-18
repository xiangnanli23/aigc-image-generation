import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from PIL.Image import Resampling
import torch
from typing import List, Union
import random
import math

import io
from io import BytesIO
import base64



RESIZE_METHOD_MAP = {
    "NEAREST": Image.NEAREST,
    "LANCZOS": Image.LANCZOS,
    "BILINEAR": Image.BILINEAR,
    "BICUBIC": Image.BICUBIC,
    "BOX": Image.BOX,
    "HAMMING": Image.HAMMING,
}


###############################################################################
def base64_to_image(base64_data: str) -> Image.Image:
    """ base64 to Image.Image """
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    return image
    
def image_to_base64(image: Image.Image) -> str:
    """ Image.Image to base64 """
    # 将 Image 对象转换为字节流
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    # 获取字节流的 base64 编码
    base64_encoded = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return base64_encoded

def image_path_to_base64(alpha_save_path):
    """ image path to base64 """
    with open(alpha_save_path, "rb") as image_file:
        image_data = image_file.read()
    base64_encoded = base64.b64encode(image_data).decode("utf-8")
    return base64_encoded

###############################################################################
def image_composition(fgr, alpha, bgr=None, output_type: str = 'pil', output_trans: bool = False):
    #TODO: support `torch.Tensor` and complete `np.array`
    """
    Supported image types: `Image.Image` and `np.array` 
    Args:
        fgr (`Image.Image` or `np.array`): foreground image
        alpha (`Image.Image` or `np.array`): alpha matte
        bgr (`Image.Image` or `np.array`, or `None`, or `int`, or `tuple`): background image, 
            `None` means white background, 
            `int` or `tuple` means color background
        output_type (`str`): output image type, `pil` or `np.array`, default is `np.array`
        output_trans (`bool`): whether to add alpha channel to the output image, whose output channel is (H, W, 4)
    Returns:
        com (`Image.Image`): composition image, 
    """
    if isinstance(fgr, Image.Image):
        fgr = np.array(fgr.convert('RGB'))
    if isinstance(alpha, Image.Image):
        alpha = np.array(alpha.convert('RGB'))
    if isinstance(bgr, Image.Image):
        bgr = np.array(bgr.convert('RGB'))

    if isinstance(alpha, np.ndarray):
        if len(alpha.shape) == 2:
            alpha = alpha[:, :, None]
        if alpha.shape[2] == 1:
            alpha = np.concatenate([alpha, alpha, alpha], axis=2)

    h, w  = fgr.shape[:2]
    fgr   = fgr[:, :, :3].astype(np.float16)
    alpha = alpha.astype(np.float16)
    
    if bgr is None:
        bgr = np.ones_like(fgr) * 255
    if isinstance(bgr, int) or isinstance(bgr, tuple):
        bgr = np.ones_like(fgr) * bgr
    if bgr.shape != fgr.shape:
        bgr = bgr[:, :, :3]
        bgr = cv2.resize(bgr, (w, h))  # nearest interpolation
        bgr = bgr.astype(np.float16)
    
    alpha = alpha / 255.0
    com = alpha * fgr + (1 - alpha) * bgr

    if output_trans:
        com = np.concatenate([com, alpha[:, :, 0][:, :, None] * 255], axis=2)

    com = com.astype(np.uint8)
    if output_type == 'pil':
        com = Image.fromarray(com) 
    return com

###############################################################################
def closest_divisible_number(n: int, divisor: int) -> int:
    """
    将整数 n 变成能够被 divisor 整除的最近的数。
    :param n: 整数 n
    :param divisor: 整数 divisor
    :return: 能够被 divisor 整除的最接近的数
    """
    if divisor == 0:
        raise ValueError("divisor 不能为 0")
    
    # 计算向下取整和向上取整的数
    lower = (n // divisor) * divisor
    upper = lower if n % divisor == 0 else lower + divisor
    
    # 比较哪个更接近
    if abs(n - lower) < abs(n - upper):
        return lower
    else:
        return upper

def divisible_resize(
    image: Image.Image, 
    divisor: int, 
    resize_method: Union[Resampling, str] = Image.LANCZOS,
    ) -> Image.Image:
    """ 
    resize image so that its width and height are divisible by divisor 
    Args:
        resize_method (`Resampling` or `str`)
            "NEAREST" / "LANCZOS" / "BILINEAR" / "BICUBIC" / "BOX" / "HAMMING"
            Image.LANCZOS / Image.BILINEAR / Image.BICUBIC / Image.NEAREST
    """
    if isinstance(resize_method, str):
        resize_method = RESIZE_METHOD_MAP[resize_method]
    
    width, height = image.size
    new_width  = closest_divisible_number(width, divisor)
    new_height = closest_divisible_number(height, divisor)

    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), resize_method)

    return image

def ratio_resize(
    image: Image.Image, 
    size: int,
    side: str = "width",
    resize_method: Union[Resampling, str] = Image.LANCZOS,
    ) -> Image.Image:
    """
    resize image so that the specified side is equal to the specified size while retaining the aspect ratio
    Args:
        size (`int`):
        side (`str`):
            "width" or "height"
        resize_method (`Resampling` or `str`):
            "NEAREST" / "LANCZOS" / "BILINEAR" / "BICUBIC" / "BOX" / "HAMMING"
            Image.LANCZOS / Image.BILINEAR / Image.BICUBIC / Image.NEAREST
    """
    size = int(size)
    width, height = image.size
    if isinstance(resize_method, str):
        resize_method = RESIZE_METHOD_MAP[resize_method]

    if side == "width":
        new_width  = size
        new_height = int(size * height / width)
    if side == "height":
        new_height = size
        new_width  = int(size * width / height)
    
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), resize_method)

    return image

def resize_image(
    image: Image.Image, 
    width: int, 
    height: int,
    resize_method: Union[Resampling, str] = Image.LANCZOS,
    ) -> Image.Image:
    """
    Args:
        resize_method (`Resampling` or `str`):
            "NEAREST" / "LANCZOS" / "BILINEAR" / "BICUBIC" / "BOX" / "HAMMING"
            Image.LANCZOS / Image.BILINEAR / Image.BICUBIC / Image.NEAREST
    """
    width  = int(width)
    height = int(height)
    if isinstance(resize_method, str):
        resize_method = RESIZE_METHOD_MAP[resize_method]
    return image.resize((width, height), resize_method)

def resize_images(
    images: List[Image.Image], 
    width: int, 
    height: int,
    resize_method: Union[Resampling, str] = Image.LANCZOS,
    ) -> List[Image.Image]:
    """
    Args:
        resize_method (`Resampling` or `str`):
            "NEAREST" / "LANCZOS" / "BILINEAR" / "BICUBIC" / "BOX" / "HAMMING"
            Image.LANCZOS / Image.BILINEAR / Image.BICUBIC / Image.NEAREST
    """
    width  = int(width)
    height = int(height)
    if isinstance(resize_method, str):
        resize_method = RESIZE_METHOD_MAP[resize_method]
    return [image.resize((width, height), resize_method) for image in images]


# deprecated
def resize_remaining_ratio(input_image: Image.Image, size: int = 512, long_side: bool = True, divide_by_eight: bool = True) -> Image.Image:
    """
    Args:
        long_side: size will be the longer side or the shorter side

    """
    size1, size2 = input_image.size
    
    if long_side:
        if size2 < size1:
            input_image = input_image.resize((size, int(size2 / size1 * size)))
        else:
            input_image = input_image.resize((int(size1 / size2 * size), size))
    else:
        if size1 < size2:
            input_image = input_image.resize((size, int(size2 / size1 * size)))
        else:
            input_image = input_image.resize((int(size1 / size2 * size), size))

    # make sure it can be divided by 8
    if divide_by_eight:
        size1, size2 = input_image.size
        H = int(size1 - size1 % 8)
        W = int(size2 - size2 % 8)
        input_image = input_image.resize((H, W))
    return input_image

# deprecated
def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


###############################################################################
def get_canny_image(input_image: Image.Image) -> Image.Image:
    image = cv2.Canny(np.array(input_image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


###############################################################################
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


###############################################################################
def image_padding(
    image: Image.Image, 
    top, bottom, left, right, 
    padding_type: str,
    ) -> Image.Image:
    """ use opencv to padding image """
    if isinstance(image, np.ndarray):
        image = image
    if isinstance(image, Image.Image):  
        image = np.array(image)
    
    if padding_type == 'edge':
        padding_type = cv2.BORDER_REPLICATE
    if padding_type == 'reflect':
        padding_type = cv2.BORDER_REFLECT_101
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, padding_type)
    padded_image = Image.fromarray(padded_image)
    return padded_image

def outpainting_preprocess(
    image: Image.Image, 
    left_width: int, right_width: int, top_width: int, bottom_width: int,
    padding_mode,
    ) -> List[Image.Image]:
    """ 
    get expanded input image and mask 
    """
    image = image.convert('RGB')

    left_width   = int(left_width)
    right_width  = int(right_width)
    top_width    = int(top_width)
    bottom_width = int(bottom_width)

    width, height   = image.size
    expanded_width  = width + left_width + right_width
    expanded_height = height + top_width + bottom_width

    # get padding image
    if padding_mode == 'black':
        padding_color = (0, 0, 0)
        expanded_image = Image.new('RGB', (expanded_width, expanded_height), padding_color)
        expanded_image.paste(image, (left_width, top_width))
    if padding_mode == 'grey':
        padding_color = (127, 127, 127)
        expanded_image = Image.new('RGB', (expanded_width, expanded_height), padding_color)
        expanded_image.paste(image, (left_width, top_width))
    if padding_mode == 'white':
        padding_color = (255, 255, 255)
        expanded_image = Image.new('RGB', (expanded_width, expanded_height), padding_color)
        expanded_image.paste(image, (left_width, top_width))
    if padding_mode == 'edge' or padding_mode == 'reflect':
        expanded_image = image_padding(image, top_width, bottom_width, left_width, right_width, padding_mode)

    # get mask
    mask = Image.new('L', (expanded_width, expanded_height), 255)
    mask.paste(0, (left_width, top_width, left_width + width, top_width + height))
    
    return expanded_image, mask

def outpainting_preprocess_ratio(
    input_image: Image.Image, 
    vertical_expand_ratio: float, 
    horizontal_expand_ratio: float,
    fusion_pixel: int, 
    padding_mode: str,
    ) -> List[Image.Image]:
    """ 
    """
    input_image  = input_image.convert('RGB')

    o_W, o_H = input_image.size
    c_W = int(horizontal_expand_ratio * o_W)
    c_H = int(vertical_expand_ratio * o_H)
    
    # padding pixels
    if padding_mode == 'black':
        padding_color = 0
        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * padding_color
    if padding_mode == 'grey':
        padding_color = 127
        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * padding_color
    if padding_mode == 'white':
        padding_color = 255
        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * padding_color
    if padding_mode == 'edge' or padding_mode == 'reflect':
        top_width    = int((c_H - o_H) / 2.0)
        bottom_width = c_H - o_H - top_width
        left_width   = int((c_W - o_W) / 2.0)
        right_width  = c_W - o_W - left_width
        expand_img = image_padding(input_image, top_width, bottom_width, left_width, right_width, padding_mode)
        expand_img = np.array(expand_img)

    original_img = np.array(input_image)
    expand_img[int((c_H - o_H) / 2.0):int((c_H - o_H) / 2.0) + o_H, int((c_W - o_W) / 2.0):int((c_W - o_W) / 2.0) + o_W, :] = original_img
    expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255

    if vertical_expand_ratio == 1 and horizontal_expand_ratio != 1:
        expand_mask[int((c_H - o_H) / 2.0):int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + fusion_pixel:int((c_W - o_W) / 2.0) + o_W - fusion_pixel, :] = 0
    elif vertical_expand_ratio != 1 and horizontal_expand_ratio != 1:
        expand_mask[int((c_H - o_H) / 2.0) + fusion_pixel:int((c_H - o_H) / 2.0) + o_H - fusion_pixel,
                    int((c_W - o_W) / 2.0) + fusion_pixel:int((c_W - o_W) / 2.0) + o_W - fusion_pixel, :] = 0
    elif vertical_expand_ratio != 1 and horizontal_expand_ratio == 1:
        expand_mask[int((c_H - o_H) / 2.0) + fusion_pixel:int((c_H - o_H) / 2.0) + o_H - fusion_pixel,
                    int((c_W - o_W) / 2.0):int((c_W - o_W) / 2.0) + o_W, :] = 0

    expand_img  = Image.fromarray(expand_img)
    expand_mask = Image.fromarray(expand_mask)
    
    return expand_img, expand_mask

def image_fusion(
    im: Image.Image, 
    im_gen: Image.Image, 
    mask: Image.Image, 
    blur_radius: int = 3,
    ) -> Image.Image:
    """ fusion input image and inpainting result """
    im     = np.asarray(im) / 255.0
    im_gen = np.asarray(im_gen) / 255.0
    mask   = mask.convert('RGB').filter(ImageFilter.GaussianBlur(radius=blur_radius))  # blur boundary
    mask   = np.asarray(mask) / 255.0
    fusion = im_gen * mask + im * (1 - mask)
    fusion = Image.fromarray(np.uint8(fusion * 255))
    return fusion

###############################################################################
def make_inpaint_condition(image: Image.Image, image_mask: Image.Image) -> torch.Tensor:
    """ for controlnet inpainting """
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


###############################################################################
def resize_to_aspect_ratio(image_path, target_ratio=(16, 9)):
    """
    读取图像，并将其调整为指定的宽高比，通过在较短边添加黑色填充（padding）来维持原图纵横比。

    参数：
    image_path (str): 图像文件路径
    target_ratio (tuple): 目标宽高比，例如 (16, 9)
    返回：
    Image: 调整尺寸后的PIL Image对象
    """

    # 读取图像
    if isinstance(image_path, str):
        im = Image.open(image_path)
    if isinstance(image_path, Image.Image):
        im = image_path

    # 获取图像的原始尺寸
    original_width, original_height = im.size

    # 计算目标宽度和高度
    target_width, target_height = target_ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > target_ratio[0] / target_ratio[1]:  # 原始宽高比大于目标宽高比，需增加高度（纵向填充）
        new_width = int(original_height * target_ratio[0] / target_ratio[1])
        new_height = original_height
    else:  # 原始宽高比小于等于目标宽高比，需增加宽度（横向填充）
        new_width = original_width
        new_height = int(original_width * target_ratio[1] / target_ratio[0])

    # 创建新的背景画布
    bg_color = (0, 0, 0)  # 黑色
    new_im = Image.new('RGB', (new_width, new_height), bg_color)

    # 将原图居中粘贴到新画布上
    paste_left = (new_width - original_width) // 2
    paste_top = (new_height - original_height) // 2
    new_im.paste(im, (paste_left, paste_top))

    return new_im

def pad_or_resize_image(img, target_size):
    # 获取图片原始尺寸
    original_width, original_height = img.size

    # 计算目标尺寸
    target_width, target_height = target_size

    # 如果目标尺寸大于图片尺寸，则在四周添加黑边
    if target_width > original_width and target_height > original_height:
        padding = ((target_width - original_width) // 2, (target_height - original_height) // 2)
        img = ImageOps.expand(img, border=padding, fill=(0, 0, 0))  # 填充黑色

    # 如果目标尺寸小于图片尺寸，则先缩放图片到目标尺寸，然后再添加黑边
    elif target_width < original_width or target_height < original_height:
        img = img.resize(target_size)  # 缩放到目标尺寸
        width_diff = target_width - img.width
        height_diff = target_height - img.height
        padding = (width_diff // 2, height_diff // 2)
        img = ImageOps.expand(img, border=padding, fill=(0, 0, 0))  # 填充黑色

    return img


###############################################################################
def concat_images(images: List[Image.Image], vertical: bool = False) -> Image.Image:
    """
    将多张图片拼接成一张图片
    Args:
        images (List[Image.Image]): 图片列表
        vertical (bool): 是否垂直拼接，默认为水平拼接
    Returns:
        Image: 拼接后的图片
    """
    array_images = [np.array(image) for image in images]
    # check size, make every image is the same size as the first one
    for i in range(1, len(array_images)):
        if array_images[i].shape != array_images[0].shape:
            array_images[i] = cv2.resize(array_images[i], (array_images[0].shape[1], array_images[0].shape[0]))
    
    # concat images
    if vertical:
        concat_image = np.concatenate(array_images, axis=0)
    else:
        concat_image = np.concatenate(array_images, axis=1)
    return Image.fromarray(concat_image)


# post processing for YSJ theater
###############################################################################
def random_crop_and_resize(img: Union[str, Image.Image], target_resolution: tuple) -> Image.Image:
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    if isinstance(img, Image.Image):
        img = img

    # 获取原图的宽度和高度
    original_width, original_height = img.size

    # 确保目标分辨率小于等于原图分辨率
    if target_resolution[0] > original_width or target_resolution[1] > original_height:
        raise ValueError("Target resolution should be smaller than or equal to the original image resolution.")

    # 随机选择裁剪起点坐标
    start_x = np.random.randint(0, original_width - target_resolution[0])
    start_y = np.random.randint(0, original_height - target_resolution[1])

    # 裁剪图片
    cropped_img = img.crop((start_x, start_y, start_x + target_resolution[0], start_y + target_resolution[1]))

    # 将裁剪后的图片重新调整回原图大小，这里假设我们希望保持原图比例，因此使用Image.ANTIALIAS进行高质量缩放
    resized_img = cropped_img.resize((original_width, original_height))

    return resized_img

def random_rotate(image, max_angle):
    # 获取输入图片的尺寸和中心坐标
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 随机生成旋转角度
    angle = np.random.uniform(0, max_angle)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # 进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def rotate_image(image: Union[str, Image.Image, np.array], max_angle: int) -> Image.Image:
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    if isinstance(image, np.ndarray):
        image = image

    # 获取输入图片的尺寸和中心坐标
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 随机生成旋转角度
    angle = np.random.uniform(0, max_angle)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # 进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    
    rotated_image = Image.fromarray(rotated_image)
    return rotated_image

def rotate_image_pil(image: Image.Image, angle: int, resample=Resampling.BICUBIC) -> Image.Image:
    image = image.convert('RGB')
    # angle = random.randint(-30, 30)
    rotated_image = image.rotate(angle, resample=resample)
    return rotated_image

def rotate_image_cv2(image: Union[str, Image.Image, np.array], ra, is_mask: bool = False) -> Image.Image: #ra旋转角度
    '''
    填充模式参数：
    BORDER_CONSTANT #恒像素值填充
    Python: cv.BORDER_CONSTANT
    iiiiii|abcdefgh|iiiiiii with some specified i

    BORDER_REPLICATE #边界像素值填充
    Python: cv.BORDER_REPLICATE
    aaaaaa|abcdefgh|hhhhhhh

    BORDER_REFLECT #翻转像素值填充
    Python: cv.BORDER_REFLECT
    fedcba|abcdefgh|hgfedcb

    BORDER_WRAP #对称像素值填充
    Python: cv.BORDER_WRAP
    cdefgh|abcdefgh|abcdefg

    BORDER_REFLECT_101 #翻转像素值（去掉边界值）填充
    Python: cv.BORDER_REFLECT_101
    gfedcb|abcdefgh|gfedcba

    BORDER_TRANSPARENT #透明填充
    Python: cv.BORDER_TRANSPAREN

    int类型的flags - 插值方法的标识符。此参数有默认值INTER_LINEAR(线性插值)，可选的插值方式如下：
    INTER_NEAREST - 最近邻插值
    INTER_LINEAR - 线性插值（默认值）
    INTER_AREA - 区域插值
    INTER_CUBIC –三次样条插值
    INTER_LANCZOS4 -Lanczos插值
    CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.
    CV_WARP_INVERSE_MAP - 表示M为输出图像到输入图像的反变换, 因此可以直接用来做象素插值。否则, warpAffine函数从M矩阵得到反变换。

    '''
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    if isinstance(image, np.ndarray):
        image = image
    # image = image.astype(np.float16)
    
    (h, w) = image.shape[:2]
    center = (w/2, h/2)
    # ra = np.random.randint(-ra, ra)
    M = cv2.getRotationMatrix2D(center, ra, 1.0) #第三个参数缩放比例
    if is_mask:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) #边界填充模式
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) #边界填充模式


    # convert to Image
    # rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    rotated = Image.fromarray(rotated)

    return rotated

def crop_center(img: Union[str, Image.Image], target_resolution: tuple) -> Image.Image:
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    if isinstance(img, Image.Image):
        img = img

    # 获取原图的宽度和高度
    original_width, original_height = img.size

    # 检查输入的分辨率是否超出了原图尺寸
    if target_resolution[0] > original_width or target_resolution[1] > original_height:
        raise ValueError("Target resolution exceeds the original image size.")

    # 计算从中心开始crop的左上角坐标
    start_x = (original_width - target_resolution[0]) // 2
    start_y = (original_height - target_resolution[1]) // 2

    # Crop图像
    cropped_img = img.crop((start_x, start_y, start_x + target_resolution[0], start_y + target_resolution[1]))

    return cropped_img


def dilate_image(img: Image.Image, iterations=1):
    dilated_img = img.filter(ImageFilter.MaxFilter(size=iterations * 2 + 1))  # MaxFilter相当于膨胀操作
    return dilated_img



def erode_image(image: Image.Image, kernel_size, iterations=1) -> Image.Image:
    # 读取图像
    img = np.array(image)
    
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 定义腐蚀核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 执行腐蚀操作
    erosion_result = cv2.erode(gray, kernel, iterations=iterations)
    
    erosion_result = Image.fromarray(erosion_result)
    return erosion_result


def rotate_crop_dilate_image(
        image: Image.Image, 
        resample: Resampling = Resampling.BICUBIC,
        angle: int = 30, 
        ratio: float = 0.8, 
        iterations: int = 1, 
        do_dilate: bool = False
    ) -> Image.Image:
    ori_w, ori_h = image.size
    image = rotate_image_pil(image, angle, resample)
    image = rotate_image_cv2(image, angle)
    w = int(ori_w * ratio)
    h = int(ori_h * ratio)
    image = crop_center(image, (w, h))
    
    if do_dilate:
        image = image.resize((ori_w, ori_h), Image.NEAREST)
        # image = erode_image(image, kernel_size=3, iterations=iterations)
        # image = dilate_image(image, iterations=iterations)
        return image
    else:
        image = image.resize((ori_w, ori_h), Image.LANCZOS)
        return image


def center_crop_and_resize(image, size):
    """
    对输入的PIL图像进行根据输入的size比例进行中心裁剪，并调整大小。

    参数:
    image (PIL.Image.Image): 输入的PIL图像
    size (tuple): 目标尺寸 (width, height)

    返回:
    PIL.Image.Image: 经过中心裁剪和调整大小后的图像
    """
    # 获取输入图像的尺寸
    width, height = image.size
    target_width, target_height = size

    # 计算目标裁剪区域的比例
    target_ratio = target_width / target_height
    current_ratio = width / height

    if current_ratio > target_ratio:
        # 原图更宽，裁剪宽度
        new_width = int(target_ratio * height)
        new_height = height
    else:
        # 原图更高，裁剪高度
        new_width = width
        new_height = int(width / target_ratio)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # 进行中心裁剪
    image = image.crop((left, top, right, bottom))
    print(image.size)

    # 调整图像大小
    image = image.resize(size)

    return image


# YSJ - NPC Batch Generation
###############################################################################
def crop_edge_remain_ratio(im: Image.Image, vertical_pixel_number: int) -> Image.Image:
    """
    Args:
        im (Image.Image): 输入图片
        pixel_number (int): 裁剪像素数
    Returns:
        Image.Image: 裁剪后的图片
    """
    width, height = im.size
    new_width = width / height * (height - vertical_pixel_number * 2)
    new_width = int(new_width)
    crop_width = int((width - new_width) // 2)
    return im.crop((crop_width, vertical_pixel_number, width - crop_width, height - vertical_pixel_number))







