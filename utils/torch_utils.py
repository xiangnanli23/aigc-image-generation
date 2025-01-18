import gc
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def clean_cache():
    """
    cost - 0.098743 s
    """
    gc.collect()
    torch_gc()

def build_generator(seed: int = -1, device=None, return_seed=False):
    if not isinstance(seed, int):  # like float
        seed = int(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if seed == -1:
        seed = np.random.randint(0, pow(2, 32))
    generator = torch.Generator(device).manual_seed(seed)
    if return_seed:
        return generator, seed
    else:
        return generator


def to_cuda(models):
    ''' move models to cuda '''
    device = "cuda"
    for model in models:  # vae, text_encoder, unet, ...
        model.to(device, dtype=torch.float16)
    return models


########################################################################
def find_missing_and_unexpected_keys(model, state_dict):
    model_state_dict = model.state_dict()
    loaded_keys = list(state_dict.keys())

    expected_keys   = list(model_state_dict.keys())
    missing_keys    = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    # return missing_keys, unexpected_keys

def find_mismatched_keys(state_dict, model_state_dict):
    """
    Args:
        state_dict (dict): the state dict to load.
        model_state_dict (dict): the model's state dict.
    """
    mismatched_keys = []
    loaded_keys = list(state_dict.keys())
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key
        if (model_key in model_state_dict) and \
           (state_dict[checkpoint_key].shape != model_state_dict[model_key].shape):
            mismatched_keys.append((checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape))
    return mismatched_keys

def compare_checkpoints(checkpoint1, checkpoint2):
    # Check if keys are the same
    keys1 = set(checkpoint1.keys())
    keys2 = set(checkpoint2.keys())

    if keys1 != keys2:
        print("Error: Checkpoints have different keys.")
        return

    # Check if values have the same shape
    for key in keys1:
        value1 = checkpoint1[key]
        value2 = checkpoint2[key]

        if value1.shape != value2.shape:
            print(f"Error: Shape mismatch for key '{key}': {value1.shape} vs {value2.shape}")
            return

    # Check if values are equal
    for key in keys1:
        value1 = checkpoint1[key]
        value2 = checkpoint2[key]

        if not torch.allclose(value1, value2, rtol=1e-5, atol=1e-8):
            print(f"Error: Values mismatch for key '{key}'")
            return

    print("Checkpoints are identical.")

########################################################################
def array2tensor(array: np.ndarray, device=None, dtype=torch.float16) -> torch.Tensor:
    """
    Args:
        array (np.ndarray):
        device (str):
    Returns:
        tensor (torch.Tensor):
    """
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tensor = torch.from_numpy(array).to(device)
    tensor = tensor / 127.5 - 1.0  # normalize to [-1, 1]
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)  # NCHW
    tensor = tensor.to(dtype)
    return tensor

def pil2tensor(image: Image.Image, device=None) -> torch.Tensor:
    array = np.array(image)
    tensor = array2tensor(array, device)
    return tensor

def tensor2list(tensor: torch.tensor) -> list:
    return tensor.tolist()

def list2tensor(python_list: list, device: str = 'cuda'):
    torch_tensor = torch.tensor(python_list, dtype=torch.float16, device=device)
    return torch_tensor


def repeat_tensor(tensor: torch.tensor, repeat_times: int) -> torch.tensor:
    repeat_dims = [1]
    if tensor.shape[0] == 2:  # do_classifier_free_guidance
        neg_tensor, tensor = tensor.chunk(2)
        tensor = tensor.repeat(repeat_times, *(repeat_dims * len(tensor.shape[1:]))
        )
        neg_tensor = neg_tensor.repeat(repeat_times, *(repeat_dims * len(neg_tensor.shape[1:])))
        tensor = torch.cat([neg_tensor, tensor])
    else:
        tensor = tensor.repeat(repeat_times, *(repeat_dims * len(tensor.shape[1:]))
        )
    return tensor


########################################################################
def tensor_pow_sum(tensor: torch.tensor, pow_base: int = 2) -> torch.tensor:
    return tensor.pow(pow_base).sum()

def normalize_tensor(tensor: torch.tensor, norm_pow_base: int = 2) -> torch.tensor:
    tensor = tensor / tensor.norm()
    return tensor



if __name__ == "__main__":
    pass


