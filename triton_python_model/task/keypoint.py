# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import triton_python_backend_utils as pb_utils

from .base import PostBaseModel


class PostKeypointDetectionModel(PostBaseModel):
    def __init__(self, input_names: List[str], output_names: List[str]) -> None:
        """Constructor function

        Args:
            input_names (List[str]): a list of model input variable names in the model configuration
            output_names (List[str]): a list of model output variable names in the model configuration
        """
        super().__init__(input_names, output_names)
        self.task = 'KEYPOINT'

    @abstractmethod
    def post_process_per_image(self, inputs: Tuple[np.ndarray]) -> List[np.ndarray]:
        """Post-process keypoint detection output in one image of a batch.

        Args:
            inputs (Tuple[np.ndarray]): a sequence of model input array for one image

        Raises:
            NotImplementedError: all subclasses must implement this function of per-image post-processing for `KEYPOINT` task.

        Returns:
            np.ndarray: keypoint detection score array `scores` for this image. The shape of `scores` should be (n,), where n is the number of categories.
        """
        raise NotImplementedError(
            f'Implement per-image inference function for {self.task} model')

    def post_process_batch_request(self, request: pb_utils.InferenceRequest) -> List[np.ndarray]:
        """Post-process keypoint detection outputs in multiple images of a Triton Inference request.

        Args:
            request (pb_utils.InferenceRequest): a Triton batch inference request

        Returns:
            np.ndarray: batched keypoint detection arrays.
        """
        inputs_list = []
        for name in self.input_names:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                raise ValueError(
                    f'Input tensor {name} not found in request {request.request_id()}')
            inputs_list.append(tensor.as_numpy())
        # This model do not support batching, so process with a single image input
        post_results = self.post_process_per_image(inputs_list)
        return [
            np.asarray(post_results[i], dtype=pb_utils.triton_string_to_numpy(self.output_configs[name]['data_type']))
            for i, name in
            enumerate(self.output_names)]
