import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Union, List
import gradio as gr

import numpy as np
from PIL import Image


from utils.time_utils import get_date


    
class GradioUIBase:
    """ abstract class for gradio ui """
    
    def __init__(self) -> None:
        pass

    def _title(self, title_name: str):
        gr.Markdown(f"<div align='center'><font size='10'>{title_name}</font></div>")
        
    def _build_tab(self, tab_name: str, program):
        with gr.Tab(tab_name): program()

    def _build_tab_select(self, tab_name: str, program):
        """ rerender ui everytime the tab is selected """
        with gr.Tab(tab_name) as tab: 
            outputs = program()
            tab.select(program, outputs=[*outputs])                

    def _button_run(self, button, button_function, inputs: list, outputs: list):
        button.click(fn=button_function, inputs=inputs, outputs=outputs)
    
    def program(self):
        pass

    def start(self, server_name, server_port, show_error=True):
        block = gr.Blocks().queue()
        with block:
            self._title("My Demo")
            self._build_tab("program 1", self.program)
            self._build_tab("program 2", self.program)
        block.launch(server_name=server_name, server_port=server_port, show_error=show_error)















def parse_gradio_image(image: dict):
    """
    image is from ImageEditor
    supported gradio version: at least 4.X.X
    Args:
        image: a dict with keys: `background`, `layers`, `composite`
    Returns:
        base_image: PIL.Image
        mask_image: PIL.Image (3 channels)
        com_image:  PIL.Image
    """
    if isinstance(image, dict):
        base_image = image['background']
        mask_image = image['layers'][0]
        com_image  = image['composite']

        base_image = base_image[:, :, :3]
        mask_image = mask_image[:, :, 3]
        mask_image = np.repeat(mask_image[:, :, None], 3, axis=2)
        com_image  = com_image[:, :, :3]

        base_image = Image.fromarray(base_image).convert('RGB')
        mask_image = Image.fromarray(mask_image).convert('RGB')
        com_image  = Image.fromarray(com_image).convert('RGB')
        return base_image, mask_image, com_image
    else:
        return None
    

def check_file_type(file_path: str, accepted_suffix: list) -> bool:
    file_suffix = file_path.split('.')[-1]
    if file_suffix in accepted_suffix or len(accepted_suffix) == 0:
        return True
    else:
        return False



def copy_uploaded_file(
    files: Union[str, List[str]],
    copy_to_dir: str = None,
    check_file: bool = False,
    **kwargs,
    ) -> Union[str, None]:
    """ 
    copy uploaded file to specific dir 
    Args:
        want_file_types: List[str]
    Returns:
        file_save_path
    """
    if isinstance(files, str):
        files = [files]
    
    if len(files) != 1:
        gr.Warning("Please just upload one file at a time!")
        return None
    
    file_path = files[0]

    file_ok = True
    if check_file:
        want_file_types = kwargs.get('want_file_types', [])
        if check_file_type(file_path, want_file_types):
            file_ok = True
        else:
            gr.Warning("Please upload a file with the correct file type!")
            file_ok = False

    if file_ok:
        file_name = os.path.basename(file_path)
        date = get_date(output_type="second")
        file_save_path = f"{copy_to_dir}/{date}_{file_name}"

        os.system(f"cp '{file_path}' '{file_save_path}'")
        
        gr.Info("Successfully upload a file!")

        return file_save_path