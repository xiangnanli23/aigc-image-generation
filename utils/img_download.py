import io
import requests
from PIL import Image


def img_download(imgUrl: str) -> Image.Image:
	"""
	功能: 推理预处理, imgUrl转为Image.Image
	"""
	def _convert_to_internal_url(url: str):
		url = url.replace("chat.cdn.soulapp.cn", "soul-chat.oss-cn-hangzhou-internal.aliyuncs.com")
		url = url.replace("china-chat-img.soulapp.cn", "soul-chat.oss-cn-hangzhou-internal.aliyuncs.com")
		url = url.replace("china-img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
		# url = url.replace("china-img.soulapp.cn", "soul-app.oss-cn-hangzhou.aliyuncs.com")
		url = url.replace("img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
		# url = url.replace("soul-app.oss-cn-hangzhou.aliyuncs.com", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
		url = url.replace("https", "http")		
		return url

	imgUrl = _convert_to_internal_url(imgUrl)
	r = requests.get(imgUrl, stream=True)
	tmp_d = next(r.iter_content(chunk_size=15))
	if b'\x00\x00\x00' in tmp_d[:10]:
		imgUrl += '?x-oss-process=image/resize,m_fill,h_5000,w_5000/format,png/quality,q_80'
	
	try:
		if str(tmp_d[:4],encoding='utf-8') == "RIFF":
			res = requests.get(imgUrl).content
			im = Image.open(io.BytesIO(res))
			imgByteArr = io.BytesIO()
			im.save(imgByteArr,"PNG")
			img_bin = imgByteArr.getvalue()
		elif str(tmp_d[:4],encoding='utf-8') != "RIFF":
			img_bin = requests.get(imgUrl).content
	except:
		img_bin = requests.get(imgUrl).content
	
	bytes_obj = io.BytesIO(img_bin)
	image = Image.open(bytes_obj).convert('RGB')
	return image


