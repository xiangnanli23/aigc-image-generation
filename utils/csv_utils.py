import os
import csv
from typing import Union, List
import pandas as pd

#########################################################################################################
def read_csv(
    file_path: str,
    mode: str = 'r',
    add_index: bool = False,
    ) -> List[dict]:
    """
    读取CSV文件并将每一行解析为字典
    output format
        [
            {'A': 'a', ...},
            {...},
            {'A': 'a', ...},
        ]
    读取CSV文件并打印每一行
    csv_reader = csv.reader(file)
    """
    with open(file_path, mode=mode, newline='', encoding='utf-8') as file:
        csv_dict_reader = csv.DictReader(file)
        if add_index:
            datas = []
            for i, row in enumerate(csv_dict_reader):
                row.update({'index': i + 1})
                datas.append(row)
            return datas
        else:
            datas = [row for row in csv_dict_reader]
            return datas


def save_csv(
    data: List[Union[dict, list]],
    fieldnames: list = None,
    save_path: str = None,
    mode: str = 'a',
    write_header: bool = False,
    ):
    """
    写入CSV文件
    e.g.
        data = [
            {'Name': 'Alice', 'Age': 30, 'Occupation': 'Engineer'},
            {'Name': 'Bob', 'Age': 25, 'Occupation': 'Data Scientist'},
            {'Name': 'Charlie', 'Age': 35, 'Occupation': 'Teacher'}
        ]
    Args:
        mode - 'a' or 'w'

    写入CSV文件
    with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)
    """
    # get fieldnames automatically
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        fieldnames = [fieldname for fieldname in data[0].keys()]
    
    if not os.path.exists(save_path):
        write_header = True

    with open(save_path, mode=mode, newline='', encoding='utf-8') as file:
        if fieldnames is not None:
            csv_dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                csv_dict_writer.writeheader()
            csv_dict_writer.writerows(data)
        else:
            csv_writer = csv.writer(file)
            csv_writer.writerows(data)


#########################################################################################################
def excel_to_csv(excel_file_path, sheet_name, csv_file_path, columns):
    # 读取Excel文件
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # 选择特定的列
    df = df[columns]
    
    # 将数据写入CSV文件
    df.to_csv(csv_file_path, index=False)


#########################################################################################################
def main_read():
    output_file = '/data2/lixiangnan/work/aigc-all/zzz_1.csv'
    
    datas = read_csv(output_file)
    print(datas[0]['score'])
    print(datas[1]['score2'])
    print(type(datas[1]['score']))
    print(type(datas[1]['score2']))
    a = float(datas[1]['score'])
    print(datas[0]['score'])
    print(a)
    print(a + 1)


def main_save():

    data = [
        {'Name': 'Charlie', 'Age': 35, 'Occupation': 'Teacher', 'score': None, 'score2': 2},
        {'Name': 'Alice', 'Age': 30, 'Occupation': 'Engineer', 'score': 1.1, 'score2': 2},
        {'Name': 'Bob', 'Age': 25, 'Occupation': 'Data Scientist', 'score': 1.1},
        
    ]
    output_file = '/data2/lixiangnan/work/aigc-all/zzz_1.csv'
    
    save_csv(data, save_path=output_file, mode='w', write_header=True)


if __name__ == "__main__":
    # input_file = '/data2/lixiangnan/work/aigc-all/test.csv'
    # datas = read_csv(input_file, add_index=True)
    # print(datas)
    # output_file = '/data2/lixiangnan/work/aigc-all/test_x.csv'
    main_read()
    # main_save()

    pass