import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
from io import BytesIO
import flask
from PIL import Image
from typing import Union

from utils.img_download import img_download
from utils.image_utils import base64_to_image, image_to_base64
from utils.exception_utils import RequestError, InferenceError
from utils.oss_utils import upload_image_url



def build_app():
    """ 
    build a flask app with health check
    """
    
    app = flask.Flask(__name__)

    @app.route("/health", methods=["GET"])
    @app.route("/health_check", methods=["GET"])
    def health():
        return "ok\n"
    
    return app


def is_url(input) -> bool:
    if isinstance(input, str) and input.startswith('http'):
        return True
    return False






def get_image(im_or_base_or_url: Union[str, Image.Image, None]) -> Union[Image.Image, None]:
    if isinstance(im_or_base_or_url, Image.Image):
        return im_or_base_or_url
    if (isinstance(im_or_base_or_url, str) and 
        (im_or_base_or_url.startswith('http') 
        or im_or_base_or_url.startswith('https'))
    ):
        try:
            return img_download(im_or_base_or_url)
        except:
            return None
    else:
        try:
            return base64_to_image(im_or_base_or_url)
        except:
            return None
    return None



def convert_image(    
    im: Union[Image.Image, str],
    return_type: str = 'url',
    ) -> Union[str, Image.Image]:
    """
    accept multiple types of image, then convert image to wanted type, like url, base64, Image.Image, etc.
    Args:
        im:
            Image.Image, url, base64, local path
        return_type:
            'url' or 'base64' or 'image'
    """
    if im is None:
        return None
    # If im is a URL, download the image
    if isinstance(im, str) and im.startswith('http'):
        im = img_download(im)
    
    # If im is a path, open the image
    elif isinstance(im, str) and os.path.isfile(im):
        im = Image.open(im)
    
    # If im is a Base64 string, decode it
    elif isinstance(im, str) and not im.startswith('http') and not os.path.isfile(im):
        im = Image.open(BytesIO(base64.b64decode(im)))

    # Now im should be an Image.Image object
    if not isinstance(im, Image.Image):
        raise ValueError("Input image format is not recognized or supported.")
    
    if return_type == "image":
        return im
    if return_type == "url":
        return upload_image_url(im)
    elif return_type == "base64":
        return image_to_base64(im)
    else:
        raise ValueError("return_type must be 'url' or 'base64' or 'image'")




def check_input(data):
    id  = data.get("id", None)
    uid = data.get("uid", None)
    model_params = data.get("model_params", None)
    extra = data.get("extra", {})

    if not isinstance(id, str):
        raise RequestError(f"RequestError: id must be str type")
    if not isinstance(uid, int):
        raise RequestError(f"RequestError: uid must be int type")
    if not isinstance(model_params, dict):
        raise RequestError(f"RequestError: model_params can't be empty")

    return id, uid, model_params, extra



def main():
    im = 'aaa'
    im = 'https://china-img.soulapp.cn/aigc_soulapp_witter/2024-06-17-aigc-witter/1f0cf869-12c4-4d9d-bb27-60ba7571e5ea.png'
    im = get_image(im)
    print(im)
    pass

def test_convert_image():
    im = 'https://china-img.soulapp.cn/aigc_soulapp_witter/2024-06-17-aigc-witter/1f0cf869-12c4-4d9d-bb27-60ba7571e5ea.png'
    im = convert_image(im)
    print(im)


if __name__ == '__main__':
    main()
    test_convert_image()
    pass