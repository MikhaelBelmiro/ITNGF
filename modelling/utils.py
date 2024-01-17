import math
import yaml
import pickle
import importlib

import numpy as np

def create_class_instance(classpath, kwargs):
    class_module_str, class_str = classpath.rsplit(".", 1)
    class_module = importlib.import_module(class_module_str)
    instance = getattr(class_module, class_str)(**kwargs)
    return instance

def read_yaml(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def load_numpy(path):
    np_array = np.load(path, allow_pickle=True)
    return np_array

def load_pickle(path):
    with open(path, "rb") as file:
        pickle_obj = pickle.load(file)
    
    return pickle_obj

def load_encoders(path_dict):
    encoder_dict = {}
    for encoder_name, encoder_path in path_dict.items():
        encoder_dict[encoder_name] = load_pickle(encoder_path)
    return encoder_dict

def get_conv_out_size(orig_size, kernel_size, dilation, stride):
    out_size = math.floor((orig_size + (-dilation) * (kernel_size - 1) - 1)/stride) + 1
    return out_size