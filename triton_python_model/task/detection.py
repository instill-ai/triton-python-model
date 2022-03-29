import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import triton_python_backend_utils as pb_utils
import triton_python_model.utils as utils


class PostDetectionModel(ABC):
    def __init__(self, input_names: List[str], output_names: List[str]) -> None:
        """Constructor function

        Args:
            input_names (List[str]): a list of model input variable names in the model configuration
            output_names (List[str]): a list of model output variable names in the model configuration
        """
        self.task = 'TASK_DETECTION'
        self.input_names = input_names
        self.output_names = output_names

        if len(output_names) != 2:
            raise ValueError(
                f'Output has {len(output_names)} elements. There should be 2 outputs: the first is bounding box outputs and the second one is label outputs')

        self.bbox_output_config = {
            "name": output_names[0],
            "data_type": "TYPE_FP32",
            "dims": [-1, 5],
        }
        self.label_output_config = {
            "name": output_names[1],
            "data_type": "TYPE_STRING",
            "dims": [-1]
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

        # Validate model configuration for TASK_DETECTION task
        # 1. The first output: bounding boxes output with dim [-1, 5] and TYPE_FP32
        bbox_config = pb_utils.get_output_config_by_name(
            model_config, name=self.bbox_output_config["name"])
        if bbox_config["dims"] != self.bbox_output_config["dims"]:
            raise ValueError(
                f'Dims for output {self.bbox_output_config["name"]} is {bbox_config["dims"]}. They should be {self.bbox_output_config["dims"]}')
        if bbox_config["data_type"] != self.bbox_output_config["data_type"]:
            raise ValueError(
                f'Data type for output {self.bbox_output_config["name"]} is {bbox_config["data_type"]}. It should be {self.bbox_output_config["data_type"]}')
        self.bbox_output_config["data_type"] = pb_utils.triton_string_to_numpy(
            self.bbox_output_config["data_type"])

        # 2. The second output: labels output with dim [-1] and TYPE_STRING
        label_config = pb_utils.get_output_config_by_name(
            model_config, name=self.label_output_config["name"])
        if label_config["dims"] != self.label_output_config["dims"]:
            raise ValueError(
                f'Dims for output {self.label_output_config["name"]} is {label_config["dims"]}. They should be {self.label_output_config["dims"]}')
        if label_config["data_type"] != self.label_output_config["data_type"]:
            raise ValueError(
                f'Data type for output {self.label_output_config["name"]} is {label_config["data_type"]}. It should be {self.label_output_config["data_type"]}')
        self.label_output_config["data_type"] = pb_utils.triton_string_to_numpy(
            self.label_output_config["data_type"])

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
            bboxes, labels = self.post_process_batch_request(req)

            # Format outputs to build an InferenceResponse
            bboxes_tensors = pb_utils.Tensor(
                self.bbox_output_config["name"], bboxes)
            labels_tensors = pb_utils.Tensor(
                self.label_output_config["name"], labels)

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            res = pb_utils.InferenceResponse(
                output_tensors=[bboxes_tensors, labels_tensors])
            responses.append(res)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    @abstractmethod
    def post_process_per_image(self, inputs: Tuple[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process objects detected in one image of a batch.

        Args:
            inputs (Tuple[np.ndarray]): Input array for detected objects in one image

        Raises:
            NotImplementedError: all subclasses must implement this function of per-image post-processing for TASK_DETECTION task.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of bounding box array and label array: (`bboxes`, `labels`).
                - `bboxes`: the bounding boxes detected in one image of a batch with shape (n,5) or (0,). The bounding box format is [x1, y1, x2, y2, score] in the original image.
                - `labels`: the labels corresponding to the bounding boxes detected in one image of a batch with shape (n,) or (0,).
                The length of `bboxes` must be the same as that of `labels`.
        """
        raise NotImplementedError(
            f'Implement per-image inference function for {self.task} model')

    def post_process_batch_request(self, request: pb_utils.InferenceRequest) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process objects detected in multiple images of a Triton Inference request.
        The output of all images in a batch must have the same size for Triton to be able to output a numpy array. Therefore, we need to fill with non-meaningful bounding boxes with coords [-1, -1, -1, -1, -1] and label '0'


        Args:
            request (pb_utils.InferenceRequest): a Triton batch inference request

        Returns:
            Tuple[np.ndarray, np.ndarray]: a Tuple of batched bounding box array and label array: (`batch_boxes`, `batch_labels`).
                - `batch_boxes`: the shape of batched bounding box array should be (batch_size, n, 5).
                - `batch_labels`: the shape of batched label array should be (batch_size, n).
        """
        inputs_list = []
        for name in self.input_names:
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            if tensor is None:
                raise ValueError(
                    f'Input tensor {name} not found in request {request.request_id()}')
            # tensor shape: (batch_size, ...)
            inputs_list.append(tensor.as_numpy())

        max_num_bboxes_in_single_image = 0
        bboxes_list, labels_list = [], []
        for inputs in zip(*inputs_list):
            # `bboxes`: the bounding boxes detected in one image of a batch with shape (n, 5) or (0,)
            # `labels`: the labels corresponding to the bounding boxes detected in one image of a batch with shape (n,) or (0,).
            # The length of `bboxes` must be the same as that of `labels`.
            bboxes, labels = self.post_process_per_image(inputs)

            if bboxes.shape[0] != labels.shape[0]:
                raise ValueError(
                    f'The length of the output bounding box array {bboxes.shape} should be the same as that of the output label array {labels.shape} for each image')

            if len(bboxes.shape) == 2:  # shape: (n, 5)  `n` is the number of bounding boxes
                if bboxes.shape[-1] != self.bbox_output_config["dims"][-1]:
                    raise ValueError(
                        f'Shape of the output bounding box array for each image should be either (n, 5) or (0,)')
            elif len(bboxes.shape) == 1:    # shape: (0,)
                if bboxes.shape[-1] != 0:
                    raise ValueError(
                        f'Shape of the output bounding box array for each image should be either (n, 5) or (0,)')
            else:
                raise ValueError(
                    f'Shape of the output bounding box array for each image should be either (n, 5) or (0,)')

            if len(labels.shape) != 1:  # shape: (n,)
                raise ValueError(
                    f'Shape of the output label array for each image should be either (n,) or (0,)')

            max_num_bboxes_in_single_image = max(
                len(bboxes), max_num_bboxes_in_single_image)

            bboxes_list.append(bboxes)
            labels_list.append(labels)

        if max_num_bboxes_in_single_image == 0:
            # When no detected object at all in all images in the batch
            for idx, _ in enumerate(zip(bboxes_list, labels_list)):
                num_to_add = 1
                bboxes_list[idx] = - np.ones(
                    (num_to_add, self.bbox_output_config["dim"][-1]), dtype=self.bbox_output_config["data_type"])
                labels_list[idx] = ["0"] * num_to_add
        else:
            for idx, (bboxes, labels) in enumerate(zip(bboxes_list, labels_list)):
                if len(bboxes) < max_num_bboxes_in_single_image:
                    num_to_add = max_num_bboxes_in_single_image - len(bboxes)
                    bboxes_to_add = - np.ones(
                        (num_to_add, self.bbox_output_config["dim"][-1]), dtype=self.bbox_output_config["data_type"])
                    labels_to_add = ["0"] * num_to_add
                    if len(bboxes) == 0:
                        bboxes_list[idx] = bboxes_to_add
                        labels_list[idx] = labels_to_add
                    else:
                        bboxes_list[idx] = np.vstack(bboxes, bboxes_to_add)
                        labels_list[idx] = labels + labels_to_add

        batch_bboxes = np.asarray(
            bboxes_list, dtype=self.bbox_output_config["data_type"])
        batch_labels = np.asarray(
            labels_list, dtype=self.label_output_config["data_type"])

        return batch_bboxes, batch_labels
