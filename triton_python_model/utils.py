from typing import Any, List, Dict
import triton_python_backend_utils as pb_utils


def validate_model_config(model_config: Dict[str, Any], input_names: List[str], output_names: List[str]) -> None:
    """Validate the model configuration based on the input and output variable names.

    Args:
        model_config (Dict[str, Any]): a JSON string containing the model configuration
        input_names (List[str]): a list of model input variable names
        output_names (List[str]): a list of model output variable names
    """
    if 'backend' not in model_config:
        raise ValueError('Backend is not defined in the model configuration')
    if model_config['backend'] != 'python':
        raise ValueError(
            'Backend is not set to python in the model configuration')

    if 'max_batch_size' not in model_config:
        raise ValueError(
            'Maximum batch size is not defined in the model configuration')
    if not isinstance(model_config['max_batch_size'], int):
        raise TypeError(
            f'Maximum batch size type {type(model_config["max_batch_size"])} is not int type in the model configuration')

    if 'input' not in model_config:
        raise ValueError("Input is not defined in the model configuration")

    # Check that the `input_names` is the same as the config input names
    model_config_input_names = [input_properties["name"]
                                for input_properties in model_config["input"]]
    if input_names != model_config_input_names:
        raise ValueError(
            f'{input_names} is not consistent with input names {model_config_input_names} in the model configuration')

    input_configs = [pb_utils.get_input_config_by_name(
        model_config, name) for name in input_names]
    for (name, cfg) in zip(input_names, input_configs):
        if cfg is None:
            raise ValueError(
                f'Input {name} is not defined in the model configuration')
        if 'name' not in cfg:
            raise ValueError(
                f'Name for input {name} is not defined in the model configuration')
        if 'dims' not in cfg:
            raise ValueError(
                f'Dims for input {name} are not defined in the model configuration')
        if not isinstance(cfg['dims'], list):
            raise TypeError(
                f'Dims for input {name} type {type(cfg["dims"])} are not list type in the model configuration')
        if 'data_type' not in cfg:
            raise ValueError(
                f'Data type for input {name} is not defined in the model configuration')

    if 'output' not in model_config:
        raise ValueError("Output is not defined in the model configuration")

    # Check that the `output_names` is the same as the config output names
    model_config_output_names = [
        output_properties["name"] for output_properties in model_config["output"]]
    if output_names != model_config_output_names:
        raise ValueError(
            f'{output_names} is not consistent with output names {model_config_output_names} in the model configuration')

    output_configs = [pb_utils.get_output_config_by_name(
        model_config, name) for name in output_names]
    for (name, cfg) in zip(output_names, output_configs):
        if cfg is None:
            raise ValueError(
                f'Output {name} is not defined in the model configuration')
        if 'name' not in cfg:
            raise ValueError(
                f'Name for output {name} is not defined in the model configuration')
        if 'dims' not in cfg:
            raise ValueError(
                f'Dims for output {name} are not defined in the model configuration')
        if not isinstance(cfg['dims'], list):
            raise TypeError(
                f'Dims for output {name} type {type(cfg["dims"])} are not list type in the model configuration')
        if 'data_type' not in cfg:
            raise ValueError(
                f'Data type for output {name} is not defined in the model configuration')
