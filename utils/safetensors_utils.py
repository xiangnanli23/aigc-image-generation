import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pprint import pprint

import json
from typing import Union

import torch
from safetensors.torch import load_file, save_file


################################################################################
def is_safetensors(path):
  return os.path.splitext(path)[1].lower() == '.safetensors'

def update_dtype(state_dict: dict, dtype):
    """ only for torch dtype """
    for k, v in state_dict.items():
        if type(v) is torch.Tensor:
            state_dict[k] = v.to(dtype)
    return state_dict

################################################################################
def state_dict_from_safetensors(ckpt_path: str, device=None) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_file(ckpt_path, device)
    return state_dict

def state_dict_from_pth(ckpt_path: str, device=None) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    return state_dict

def state_dict_from_ckpt(ckpt_path: str, device=None) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_safetensors(ckpt_path):
        state_dict = state_dict_from_safetensors(ckpt_path, device)
    else:
        state_dict = state_dict_from_pth(ckpt_path, device)
    return state_dict

def load_checkpoint(checkpoint_path: str, 
                    device: str = None, 
                    dtype=torch.float16) -> dict:
    checkpoint = state_dict_from_ckpt(checkpoint_path, device)
    if dtype is not None:
        checkpoint = update_dtype(checkpoint, dtype)
    return checkpoint

################################################################################
class SafetensorsUtils:
    ''' Safetensors utils '''
    @staticmethod
    def state_dict_from_safetensors(ckpt_path: str) -> dict:
        """ get state dict from safetensors """
        state_dict = load_file(ckpt_path, "cpu")
        return state_dict

    @staticmethod
    def update_state_dict(tensor_path: str, tensor_base_path: str) -> dict:
        """ update state dict """
        dict_all  = SafetensorsUtils.state_dict_from_safetensors(tensor_path)
        dict_base = SafetensorsUtils.state_dict_from_safetensors(tensor_base_path)
        dict_new = {}
        for k, v in dict_all.items():
            if k in dict_base.keys():
                dict_new[k] = v
        return dict_new

    @staticmethod
    def save_safetensors(tensors, save_path: str):
        save_file(tensors, save_path)

class CkptUtils:
    @staticmethod
    def state_dict_from_ckpt(ckpt_path: str) -> dict:
        """ get state dict from ckpt """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        return checkpoint['state_dict']





################################################################################
if __name__ == '__main__':
    pass


