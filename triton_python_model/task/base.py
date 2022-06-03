# -*- coding: utf-8 -*-

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Union

import numpy as np
import triton_python_backend_utils as pb_utils


class PostBaseModel(ABC):
    def __init__(self, input_names: List[str], output_names: List[str]) -> None:
        """Constructor function

        Args:
            input_names (List[str]): a list of model input variable names in the model configuration
            output_names (List[str]): a list of model output variable names in the model configuration
        """
        self.input_names = input_names
        self.output_names = output_names
        self.input_configs = {}
        self.output_configs = {}

    def initialize(self, args: Dict[str, str]) -> None:
        """`initialize` is called only once when the model is being loaded. Implementing `initialize` function is optional.
        This function allows the model to initialize any state associated with this model.

        Args:
            args (Dict[str, str]): both keys and values are strings. The dictionary keys and values are:
                * model_config: a JSON string containing the model configuration
                * model_instance_kind: a string containing model instance kind
                * model_instance_device_id: a string containing model instance device ID
                * model_repository: model repository path
                * model_version: model version
                * model_name: model name
        """
        # Read config file
        model_config = json.loads(args['model_config'])
        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')
        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')

        self.input_configs = {name: pb_utils.get_input_config_by_name(
            model_config, name) for name in self.input_names}
        for k, cfg in self.input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        self.output_configs = {name: pb_utils.get_output_config_by_name(
            model_config, name) for name in self.output_names}
        for k, cfg in self.output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

    def execute(self, inference_requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """`execute` must be implemented in every Python model. `execute` function receives
        a list of pb_utils.InferenceRequest as the only argument.
        This function is called when an inference is requested for this model.
        """
        responses = []
        for req in inference_requests:
            post_results = self.post_process_batch_request(req)
            output_tensors = [pb_utils.Tensor(self.output_names[i], output) for i, output in enumerate(post_results)]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            res = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(res)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass

    @abstractmethod
    def post_process_batch_request(self, request: pb_utils.InferenceRequest) -> Union[np.ndarray, List[np.ndarray]]:
        """Post-process keypoint detection outputs in multiple images of a Triton Inference request.

        Args:
            request (pb_utils.InferenceRequest): a Triton batch inference request

        Raises:
            NotImplementedError: all subclasses must implement this function of per-image post-processing.
        Returns:
            np.ndarray: batched results.
        """
        raise NotImplementedError(
            f'Implement per-image inference function for {self.task} model')
