import inspect



def get_signature_parameters_init(obj) -> inspect.Parameter:
    parameters = inspect.signature(obj.__init__).parameters
    return parameters

def get_signature_parameters_call(obj) -> inspect.Parameter:
    parameters = inspect.signature(obj.__call__).parameters
    return parameters

def get_signature_keys(obj, 
                       method: str = 'init', 
                       return_all: bool = False
    ):
    """
    return expected_modules & optional_kwargs of an object's specific method
    Args:
        obj: object
        method (`str`): 'init' or 'call'
    Returns:
        expected_modules: set
        optional_kwargs: set
    """
    if method == 'init':
        parameters = get_signature_parameters_init(obj)
    if method == 'call':
        parameters = get_signature_parameters_call(obj)
    
    if return_all:
        all_modules = set(parameters.keys()) - {"self"}
        return all_modules
    else:
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}
        return expected_modules, optional_parameters



def example():
    from pprint import pprint
    from diffusers import StableDiffusionPipeline

    expected_modules, optional_kwargs = get_signature_keys(StableDiffusionPipeline, 'init')
    pprint(expected_modules)
    pprint(optional_kwargs)

    expected_modules, optional_kwargs = get_signature_keys(StableDiffusionPipeline, 'call')
    pprint(expected_modules)
    pprint(optional_kwargs)

    all_modules = get_signature_keys(StableDiffusionPipeline, 'call', return_all=True)
    pprint(all_modules)


if __name__ == '__main__':
    example()
    pass

