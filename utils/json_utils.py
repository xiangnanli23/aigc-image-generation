"""
    null in json file is None(`NoneType`) in python !!!
"""

import os
import json
from typing import Union

from pprint import pprint


####################################################################################################
def load_json(json_file: Union[str, os.PathLike]) -> dict:
    """ load json file into dict """
    assert os.path.splitext(json_file)[1].lower() == '.json', f"{json_file} is not a json file!"
    with open(json_file, "r", encoding="utf-8") as json_file:
        dict = json.load(json_file)
    return dict

def save_to_json(data: dict, save_path: str, indent: int = None) -> None:
    """ save dict to json file """
    # 将字典转换为JSON格式的字符串
    # json_data = json.dumps(data, indent=4, ensure_ascii=False)  # indent参数可选，用于美化输出（增加缩进）
    json_data = json.dumps(data, indent=indent, ensure_ascii=False)  # indent参数可选，用于美化输出（增加缩进）

    # 将JSON数据保存到文件中
    assert save_path.endswith('.json'), f"{save_path} is not a json file!"
    with open(save_path, "w",  encoding="utf-8") as json_file:
        json_file.write(json_data)

def add_to_json(json_file: Union[str, os.PathLike], data: dict) -> None:
    """ append new dict to json file """
    with open(json_file, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


####################################################################################################
def get_items(wanted_keys, data_dict) -> dict:
    config = {}
    for k, v in data_dict.items():
        if k in wanted_keys:
            config[k] = v
    return config


def replace_single_quotes_to_double(input_text):
    return input_text.replace("'", "\"")

def replace_all_single_quotes_in_a_file(input_filename, output_filename=None):
    with open(input_filename, 'r') as input_file:
        content = input_file.read()
        fixed_content = replace_single_quotes_to_double(content)

    # 如果没有指定输出文件名，则默认覆盖原文件
    if output_filename is None:
        output_filename = input_filename

    with open(output_filename, 'w') as output_file:
        output_file.write(fixed_content)

def read_multiple_jsons_from_file(file_path) -> list:
    objects_list = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # 检查每一行是否是完整的JSON对象
        for line in lines:
            trimmed_line = line.strip()
            
            # 忽略空白行和注释行
            if not trimmed_line or trimmed_line.startswith('//') or trimmed_line.startswith('#'):
                continue
            
            try:
                # 尝试将这一行解析为JSON对象
                obj = json.loads(trimmed_line)
                objects_list.append(obj)
            except json.JSONDecodeError:
                # 如果当前行不是一个完整的JSON对象，可能需要进一步处理或忽略
                print(f"Line is not a valid JSON object: {trimmed_line}")

    return objects_list
