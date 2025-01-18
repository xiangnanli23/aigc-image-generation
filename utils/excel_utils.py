"""
pandas
1. if nothing is provided, the type will be np.nan  - NULL


"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List
import numpy as np
import pandas as pd
from pandas import DataFrame
from pprint import pprint

from utils.json_utils import load_json


########################################################################
def read_excel(
    excel_file: Union[str, DataFrame],
    to_list: bool = False, 
    dict_orient: str = 'records',
    ) -> Union[DataFrame, List[dict], dict]:
    """
    Args:
        dict_orient: 
            'dict' (default) : dict like {column -> {index -> value}}
            'list' : dict like {column -> [values]}
            'series' : dict like {column -> Series(values)}
            'split' : dict like {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            'tight' : dict like {'index' -> [index], 'columns' -> [columns], 'data' -> [values], 'index_names' -> [index.names], 'column_names' -> [column.names]}
            'records' : list like [{column -> value}, ... , {column -> value}]
            'index' : dict like {index -> {column -> value}}
    """
    if isinstance(excel_file, str):
        df = pd.read_excel(excel_file, engine='openpyxl')
    if isinstance(excel_file, DataFrame):
        df = excel_file
    if to_list:
        return df.to_dict("records")
    if (not to_list) and dict_orient:
        return df.to_dict(dict_orient)
    return df

def save_to_excel(df: Union[dict, DataFrame], excel_file: str) -> None:
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    df.to_excel(excel_file, index=False)

def read_excel_into_dict(excel_file: Union[str, DataFrame]) -> List[dict]:
    """ deprecated, use read_excel instead """
    if isinstance(excel_file, str):
        # df = pd.read_excel(excel_file)
        df = pd.read_excel(excel_file, engine='openpyxl')
    if isinstance(excel_file, DataFrame):
        df = excel_file
    data_list = df.to_dict('records')
    return data_list

    # 打印转换后的数据
    # for row_dict in data_dicts:
        # print(row_dict)

def json_to_excel(json_data: dict, excel_file: str) -> None:
    if isinstance(json_data, str):
        json_data = load_json(json_data)
    df = pd.DataFrame(json_data)
    df.to_excel(excel_file, index=False)


########################################################################
def read_csv(excel_file: str, to_list: bool = False) -> Union[DataFrame, List[dict]]:
    df = pd.read_csv(excel_file)
    if to_list:
        return df.to_dict('records')
    return df


def csv_to_excel(csv_file_path, excel_file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 将DataFrame写入Excel文件
        df.to_excel(excel_file_path, index=False)
        
        print(f"成功将CSV文件转换为Excel文件：{excel_file_path}")
    except Exception as e:
        print(f"转换过程中发生错误：{e}")



########################################################################
def write_to_txt(file_path, result):
    with open(file_path, 'a') as f:
        f.write(f"{result}\n")









########################################################################
def example_read_excel():
    # read excel to df
    excel_path = '/data2/lixiangnan/work/aigc-all/z.xlsx'
    df = read_excel(excel_path, dict_orient='list')
    print(df)

    for id in df['id']:
        print(type(id))
    
    # # total column count
    # column_count = df.shape[1]
    # print(f"column_count: {column_count}")

    # # total row count (do not include header)
    # row_count = df.shape[0]
    # print(f"row_count: {row_count}")

    # # read every rows
    # for idx, row in df.iterrows():
    #     print(idx)
    #     prompt = row['prompt']
    #     if prompt is np.nan:
    #         print('this is empty')
    #     else:
    #         print(row['prompt'])
    #     # for col in row:
    #         # print(col)
        
    #     # print(row.to_dict())
    #     # print("列名：", list(df.columns))

def example_read_excel_into_dict():
    # 指定 Excel 文件路径
    excel_file = "/data2/lixiangnan/work/aigc-all/test.xlsx"
    excel_file = "/data2/lixiangnan/work/aigc-all/test_2.xlsx"

    # 读取 Excel 文件的第一个工作表
    # 如果需要读取特定的工作表，可以指定 sheet_name 参数
    df = pd.read_excel(excel_file)
    print(type(df))
    if isinstance(df, DataFrame):
        print('yes')

    # 将 DataFrame 转换为字典列表，每一项都是一个字典，代表一行数据
    data_dicts = df.to_dict('records')

    # 打印转换后的数据
    for row_dict in data_dicts:
        pprint(row_dict)

    for row_dict in data_dicts:
        for k,v in row_dict.items():
            if v is np.nan:
                print('this is empty')
            # print(k, v)


    # print(data_dicts.keys())

def example_json_to_excel():
    json_path  = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/pgc_to_excel.json'
    excel_path = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/pgc_to_excel.xlsx'

    json_path  = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/ugc_female_v2_to_excel.json'
    excel_path = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/ugc_female_v2_to_excel.xlsx'
    
    json_path  = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/ugc_male_v2_to_excel.json'
    excel_path = '/data6/lixiangnan/projects/ysj_theater_v22_ugc/avatars/ugc_male_v2_to_excel.xlsx'

    json_to_excel(json_path, excel_path)

def example_read_csv():
    csv_path = '/data2/lixiangnan/work/aigc-all/Cleaned_Talkie-兆华-0425.csv'
    df = read_csv(csv_path, to_list=False)
    pprint(df)

def example_csv_to_excel():
    csv_path = '/data2/lixiangnan/work/aigc-all/批量人设_2npc.csv'
    df = read_csv(csv_path)
    excel_save_path = '/data2/lixiangnan/work/aigc-all/批量人设_2npc.xlsx'
    df.to_excel(excel_save_path, sheet_name="Testing", index=False)


def example_save_dict_to_excel():
    data_dict = {
        'id': [1, 2, ["1", "2", "3"]],
        'name': ['a', 'b', 'c'],
    }

    df = pd.DataFrame(data_dict)
    
    excel_file = '/data2/lixiangnan/work/aigc-all/z.xlsx'
    df.to_excel(excel_file, index=False)


    pass

def example_add_item_to_excel():
    excel_path = '/data2/lixiangnan/work/aigc-all/z.xlsx'
    datas = read_excel(excel_path, dict_orient='list')
    print(datas)
    print(datas.keys())
    print(hasattr(datas.keys(), 'id'))
    print('id' in datas.keys())
    
    # new_index = len(datas['id'].keys())
    # datas['id'][new_index] = 4
    # datas['name'][new_index] = 'd'

    # df = pd.DataFrame(datas)
    # excel_file = '/data2/lixiangnan/work/aigc-all/z2.xlsx'
    # df.to_excel(excel_file, index=False)

    # print(datas)


########################################################################
if __name__ == '__main__':
    example_read_excel()
    # example_read_excel_into_dict()
    # example_json_to_excel()
    # example_read_csv()
    # example_save_dict_to_excel()
    # example_csv_to_excel()
    # example_add_item_to_excel()
    pass