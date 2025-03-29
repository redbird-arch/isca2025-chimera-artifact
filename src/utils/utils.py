# File name  :    utils.py
# Author     :    xiaocuicui
# Time       :    2023/10/12 15:38:14
# Version    :    V1.0
# Abstract   :    this file contains functions for reading file and writing file

import configparser
import pickle
import json
import requests
import os
file_path = os.path.dirname(os.path.realpath(__file__))

def print_list(lst: list):
    for k in lst:
        print(k)

def print_dict(dct: dict):
    for k, v in dct.items():
        print(k)


def read_cfg(file_path):
    """
    Read a .cfg file with INI format and perform automatic type conversion for
    integers, floating-point numbers, booleans, and strings.
    
    :param file_path: Path to the configuration file.
    :return: A dictionary where each key represents a section, 
             and each value is a dictionary of key-value pairs within that section.
    """

    if not os.path.exists(file_path):
        raise ValueError("No such file or directory: " + file_path)
    if not file_path.endswith('.cfg'):
        raise ValueError("The file_path should have a '.cfg' extension for config format!")    
    config = configparser.ConfigParser()
    config.read(file_path)

    config_data = {}
    for section in config.sections():
        config_data[section] = {}
        for key in config[section]:
            value = config[section][key]
            # Convert string values to corresponding data types
            if value.isdigit():
                config_data[section][key] = int(value)
            elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                config_data[section][key] = float(value)
            elif value.lower() in ['true', 'false']:
                config_data[section][key] = value.lower() == 'true'
            else:
                config_data[section][key] = value

    return config_data


def save_dict_json(data, filename):
    """
    save dict or list to json file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_dict_json(filename):
    """
    load dict or list from json file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_dict_pickle(data, filename):
    """
    save dict or list to pkl file
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_dict_pickle(filename):
    """
    load dict or list from pkl file
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data



'''
save and load list to local file
'''
def save_list_pickle(lst, file_path):
    if not file_path.endswith('.pkl'):
        raise ValueError("The file_path should have a '.pkl' extension for pickle format!")
    with open(file_path, 'wb') as f:
        pickle.dump(lst, f)

def load_list_pickle(file_path):
    if not os.path.exists(file_path):
        raise ValueError("No such file or directory: " + file_path)    
    if not file_path.endswith('.pkl'):
        raise ValueError("The file_path should have a '.pkl' extension for pickle format!")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_list_txt(lst, file_path):
    if not file_path.endswith('.txt'):
        raise ValueError("The file_path should have a '.txt' extension for text format!")
    with open(file_path, 'w') as f:
        for item in lst:
            f.write(f"{item}\n")

def load_list_txt(file_path):
    if not os.path.exists(file_path):
        raise ValueError("No such file or directory: " + file_path)    
    if not file_path.endswith('.txt'):
        raise ValueError("The file_path should have a '.txt' extension for text format!")
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def save_list_json(lst, file_path):
    if not file_path.endswith('.json'):
        raise ValueError("The file_path should have a '.json' extension for JSON format!")
    with open(file_path, 'w') as f:
        json.dump(lst, f)

def load_list_json(file_path):
    if not os.path.exists(file_path):
        raise ValueError("No such file or directory: " + file_path)    
    if not file_path.endswith('.json'):
        raise ValueError("The file_path should have a '.json' extension for JSON format!")
    with open(file_path, 'r') as f:
        return json.load(f)


'''
download file from url
'''

def download_file(url, dest_path):
    """
    Download a file from the specified URL and save it to the provided destination path.
    While downloading, connection status and download progress will be printed.
    
    :param url: URL of the file to be downloaded.
    :param dest_path: Complete local path (including filename) where the file should be saved.
    """
    response = requests.get(url, stream=True)
    
    # Check connection status
    if response.status_code == 200:
        print("Connected to the URL successfully!")
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        downloaded_size = 0
        
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                downloaded_size += len(data)
                file.write(data)
                progress = (downloaded_size / total_size) * 100
                print(f"Downloaded: {downloaded_size} of {total_size} bytes ({progress:.2f}%)")
    else:
        print(f"Failed to connect to the URL. HTTP Status Code: {response.status_code}")


'''
deal with units
'''
def convert_to_bytes(value, unit):
    """
    Convert a value with a given unit to bytes.
    
    :param value: Value to be converted.
    :param unit: Unit of the value.
    :return: Value in bytes.
    """
    if unit == '':
        return value
    elif unit == 'k' or unit == 'K':
        return value * 1024
    elif unit == 'm' or unit == 'M':
        return value * 1024 * 1024
    elif unit == 'g' or unit == 'G':
        return value * 1024 * 1024 * 1024
    elif unit == 't' or unit == 'T':
        return (value * 1024 * 1024 * 1024 * 1024)
    else:
        raise ValueError(f"Invalid unit: {unit}. Please use B, KB, MB, GB, or TB.")

def convert_to_seconds(value, unit):
    """
    Convert a value with a given unit to seconds.
    
    :param value: Value to be converted.
    :param unit: Unit of the value.
    :return: Value in bytes.
    """
    if unit == 's':
        return value
    elif unit == 'ms':
        return value / 1000
    elif unit == 'us':
        return value / 1000 / 1000
    elif unit == 'ns':
        return value / 1000 / 1000 / 1000
    elif unit == 'ps':
        return value / 1000 / 1000 / 1000 / 1000
    else:
        raise ValueError(f"Invalid unit: {unit}. Please use s, ms, us, ns or ps.")

'''
get model json
'''

def get_gpt2_config(model_name, config_path):
    '''
    ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/models/gpt2/configuration_gpt2.py

    Example:
    >>> gpt2_dict = gpt2.get_gpt_parameter('gpt2-medium')
    '''
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
    "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
    "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/config.json",
    }

    # config_path = os.path.join(file_path, f'../input/model/{model_name}-config.json')
    if os.path.exists(config_path) is False:
        # download_file(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name], config_path)
        raise ValueError("No such file or directory: " + config_path)
    else:
        pass
    model_config = load_dict_json(config_path)

    return model_config


def get_network_config(folder_path):

    return read_cfg(os.path.join(file_path, f'../input/network/{folder_path}.cfg'))


def get_task_config(task_name):

    return read_cfg(os.path.join(file_path, f'../input/task/{task_name}.cfg'))


def create_exp_folder(current_path):
    '''
    create a new folder for experiment

    '''

    max_exp_num = -1
    for item in os.listdir(current_path):  # 遍历当前目录
        if os.path.isdir(os.path.join(current_path, item)) and item.startswith('exp'):
            try:
                num = int(item[3:])  # 尝试获取数字部分
                max_exp_num = max(max_exp_num, num)
            except ValueError:
                continue  # 如果不是数字，继续遍历

    new_folder = os.path.join(current_path, f'exp{max_exp_num + 1}')
    os.mkdir(os.path.join(current_path, new_folder))
    print(f"Folder '{new_folder}' created.")

    return new_folder


def create_comm_folder(current_path):
    '''
    create a new folder for communication only

    '''

    max_comm_num = -1
    for item in os.listdir(current_path): 
        if os.path.isdir(os.path.join(current_path, item)) and item.startswith('comm'):
            try:
                num = int(item[4:]) 
                max_comm_num = max(max_comm_num, num)
            except ValueError:
                continue 

    new_folder = os.path.join(current_path, f'comm{max_comm_num + 1}')
    os.mkdir(os.path.join(current_path, new_folder))
    print(f"Folder '{new_folder}' created.")

    return new_folder
