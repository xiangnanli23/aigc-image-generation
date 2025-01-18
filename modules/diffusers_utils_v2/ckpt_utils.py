import os
import torch
from safetensors.torch import load_file


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
    while "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict

def state_dict_from_pth(ckpt_path: str, device=None) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(ckpt_path, map_location=device)
    while "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict

def state_dict_from_ckpt(ckpt_path: str, device=None) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_safetensors(ckpt_path):
        state_dict = state_dict_from_safetensors(ckpt_path, device)
    else:
        state_dict = state_dict_from_pth(ckpt_path, device)
    return state_dict

def load_checkpoint(
    checkpoint_path: str, 
    device: str = None, 
    dtype=torch.float16,
    ) -> dict:
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = state_dict_from_ckpt(checkpoint_path, device)
    if dtype is not None:
        checkpoint = update_dtype(checkpoint, dtype)
    return checkpoint




if __name__ == '__main__':
    pass


