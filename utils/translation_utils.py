import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import translators as ts
from typing import Union
import re

from utils.http_utils import HTTP


# print(ts.translators_pool)
TRANSLATORS = ts.translators_pool
SUPPORTED_LANGUAGES = ['en', 'zh', 'fr', 'de', 'es', 'ja', 'ko']




def contains_chinese(text: str) -> bool:
    # 定义一个正则表达式模式，匹配中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    # 使用正则表达式搜索字符串
    match = chinese_pattern.search(text)
    
    # 如果找到匹配项，则返回 True，否则返回 False
    return match is not None


############################################################################################################
def get_translation(
        query_text: str,
        translator: str = 'bing',
        from_language: str = 'auto',
        to_language: str = 'en',
        if_use_preacceleration: bool = False,
        timeout: int = 2,
    ) -> str:
    """
    open-source library to call translation APIs
    https://github.com/UlionTse/translators?tab=readme-ov-file
    """
    res = ts.translate_text(
        query_text=query_text, 
        translator=translator, 
        from_language=from_language,
        to_language=to_language, 
        timeout=timeout,
    )
    return res



############################################################################################################
def trans_cn2en_retry(prompt: str, max_retries: int = 3, delay: float = 0.5) -> Union[str, None]:
    """
    Translate Chinese to English with retry mechanics
    """
    prompt_en = None
    for i in range(max_retries):
        try:
            if prompt_en is not None:
                return prompt_en
            translator = random.choice(['alibaba', 'bing', 'sogou', 'youdao', 'qqTranSmart'])
            prompt_en  = get_translation(prompt, translator=translator, from_language='zh', to_language='en')
            print(translator)
        except:
            time.sleep(delay)
    return prompt_en

