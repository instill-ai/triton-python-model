import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import triton_python_backend_utils as pb_utils
import triton_python_model.utils as utils


class PostClassificationModel(ABC):
    def __init__(self, input_names: List[str], output_names: List[str]) -> None:
        """Constructor function

        Args:
            input_names (List[str]): a list of model input variable names in the model configuration
            output_names (List[str]): a list of model output variable names in the model configuration
        """
        self.task = 'CLASSIFICATION'
        self.input_names = input_names
        self.output_names = output_names

        if len(output_names) != 1:
            raise ValueError(
                f'Output has {len(output_names)} elements. There should be 1 output: score outputs')

        self.score_output_config = {
            "name": output_names[0],
            "data_type": "TYPE_FP32",
            "dims": [None]  # [n] n is the number of categories
        }

    def initialize(self, args: Dict[str, str]) -> None:
        """`initialize` is called only once when the model is being loaded. Implementing `initialize` function is optional. This function allows the model to initialize any state associated with this model.

        Args:
            args (Dict[str, str]): Both keys and values are strings. The dictionary keys and values are:
                * model_config: A JSON string containing the model configuration
                * model_instance_kind: A string containing model instance kind
                * model_instance_device_id: A string containing model instance device ID
                * model_repository: Model repository path
                * model_version: Model version
                * model_name: Model name
        """
        # Read config file
        model_config = json.loads(args['model_config'])

        # Validate general model configuration
        utils.validate_model_config(
            model_config, self.input_names, self.output_names)

        # Validate model configuration for CLASSIFICATION task
        score_config = pb_utils.get_output_config_by_name(
            model_config, name=self.score_output_config["name"])
        # 1. score must have `label_filename` in the configuration
        if 'label_filename' not in score_config:
            raise ValueError(
                f'Label filename `label_filename` for output {score_config["name"]} is not defined in the model configuration')

        # 2. score output with dim [n] and TYPE_FP32
        if len(score_config["dims"]) != 1:
            raise ValueError(
                f'Dims for output {self.score_output_config["name"]} are {score_config["dims"]}. Its length should be 1.')
        self.score_output_config["dims"] = score_config["dims"]

        if score_config["data_type"] != self.score_output_config["data_type"]:
            raise ValueError(
                f'Data type for output {self.score_output_config["name"]} is {score_config["data_type"]}. It should be {self.score_output_config["data_type"]}')
        self.score_output_config["data_type"] = pb_utils.triton_string_to_numpy(
            self.score_output_config["data_type"])

    def execute(self, inference_requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """`execute` must be implemented in every Python model. `execute` function receives a list of pb_utils.InferenceRequest as the only argument. This function is called when an inference is requested for this model.

        Args:
            inference_requests (List[pb_utils.InferenceRequest]): A list of Triton Inference Request

        Returns:
            List[pb_utils.InferenceResponse]: A list of Triton Inference Response. The length of this list must
            be the same as `inference_requests`
        """
        responses = []
        for req in inference_requests:
            scores = self.post_process_batch_request(req)

            # Format outputs to build an InferenceResponse
            scores_tensors = pb_utils.Tensor(
                self.score_output_config["name"], scores)

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            res = pb_utils.InferenceResponse(
                output_tensors=[scores_tensors])
            responses.append(res)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    @abstractmethod
    def post_process_per_image(self, inputs: Tuple[np.ndarray]) -> np.ndarray:
        """Post-process classification output in one image of a batch.

        Args:
            inputs (Tuple[np.ndarray]): Input array for detected objects in one image

        Raises:
            NotImplementedError: all subclasses must implement this function of per-image post-processing for `CLASSIFICATION` task.

        Returns:
            np.ndarray: classification score array for one image `scores`. The shape of `scores` should be (n,), where n is the number of categories.
        """
        raise NotImplementedError(
            f'Implement per-image inference function for {self.task} model')

    def post_process_batch_request(self, request: pb_utils.InferenceRequest) -> np.ndarray:
        """Post-process classification outputs in multiple images of a Triton Inference request.

        Args:
            request (pb_utils.InferenceRequest): a Triton batch inference request

        Returns:
            np.ndarray: batched classification array `batch_scores`. The shape of batched classification array should be (batch_size, n), where n is the number of categories.
        """
        inputs_list = []
        for name in self.input_names:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                raise ValueError(
                    f'Input tensor {name} not found in request {request.request_id()}')
            # tensor shape: (batch_size, ...)
            inputs_list.append(tensor.as_numpy())

        scores_list = []
        for inputs in zip(*inputs_list):
            # The scores corresponding to all categories in one image of a batch with shape (n,), where n is the number of category.
            scores = self.post_process_per_image(inputs)

            # shape: (n,)
            if not(len(scores.shape) == 1 and scores.shape[0] == self.score_output_config["dims"][0]):
                raise ValueError(
                    f'Shape of the output score array for each image is {scores.shape} and it should be {self.score_output_config["dims"]}')

            scores_list.append(scores)

        batch_scores = np.asarray(
            scores_list, dtype=self.score_output_config["data_type"])

        return batch_scores
