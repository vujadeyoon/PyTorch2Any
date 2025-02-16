"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_quic.py
Description: A module for Qualcomm Innovation Center

Reference:
    i)  https://aihub.qualcomm.com/get-started#postinstall
    ii) https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html#qai-hub-submit-compile-job
"""


import numpy as np
import qai_hub as hub
from typing import Union
from vujade import vujade_path as path_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd, pprintd


class QualcommAIHub(object):
    @staticmethod
    def get_ndarr_type(_name: str) -> type:
        dict_type = {
            'float32': np.float32,
            'int32': np.int32,
            'int16': np.int16,
            'int8': np.int8,
            'uint16': np.uint16,
            'uint8': np.uint8,
        }
        return dict_type[_name]

    @staticmethod
    def print_list_devices() -> None:
        utils_.SystemCommand.run(_command='qai-hub list-devices', _is_daemon=False, _is_subprocess=True)

    @classmethod
    def upload_model(cls, _spath_file: str) -> hub.client.Model:
        model = hub.upload_model(_spath_file)
        return model

    @classmethod
    def compile(cls, model: Union[hub.client.Model, str], **kwargs) -> hub.client.Model:
        compile_job = hub.submit_compile_job(
            model=model,
            **kwargs
        )
        model_compiled = compile_job.get_target_model()
        return model_compiled

    @classmethod
    def quantization(cls, _model: hub.client.Model, _calibration_data: dict, _weight_dtype, _activations_dtype, _device: hub.client.Device, _options: str) -> hub.client.Model:
        quantize_job = hub.submit_quantize_job(
            model=_model,
            calibration_data=_calibration_data,
            weights_dtype=_weight_dtype,
            activations_dtype=_activations_dtype,
        )
        model_onnx_quantized = quantize_job.get_target_model()

        model_quantized_compiled = cls.compile(
            model=model_onnx_quantized,
            device=_device,
            options=_options
        )
        return model_quantized_compiled


    @classmethod
    def profile(cls, _target_model: hub.client.Model, _device: hub.client.Device) -> None:
        profile_job = hub.submit_profile_job(
            model=_target_model,
            device=_device
        )

    @classmethod
    def inference(cls, _target_model: hub.client.Model, _device: hub.client.Device, _inputs: dict) -> hub.client.InferenceJob:
        inference_job = hub.submit_inference_job(
            model=_target_model,
            device=_device,
            inputs=_inputs
        )
        return inference_job

    @classmethod
    def download_output(cls, _inference_job: hub.client.InferenceJob) -> dict:
        return _inference_job.download_output_data()

    @classmethod
    def download_target_model(cls, _model_compiled: hub.client.Model, _spath_file: str) -> None:
        path_file = path_.Path(_spath_file)
        path_file.parent.path.mkdir(mode=0o775, parents=True, exist_ok=True)
        _model_compiled.download(path_file.str)
        if path_file.path.is_file():
            printd('File path: {}'.format(path_file.str), _is_pause=False)
        else:
            raise FileNotFoundError()
