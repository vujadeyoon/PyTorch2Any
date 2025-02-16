import os
import argparse
import torch
import numpy as np
import qai_hub as hub
from lib.external.dnn.InternVideo.InternVideo2.single_modality.models.internvideo2_distill import distill_internvideo2_small_patch14_224
from vujade import vujade_onnx as onnx_
from vujade import vujade_path as path_
from vujade import vujade_profiler as profiler_
from vujade import vujade_quic as quic_
from vujade.vujade_debug import printd, pprintd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='DNN model name')
    parser.add_argument('--quai_device', type=str, default='Snapdragon 8 Elite QRD', help='Qualcomm AI Hub device')
    parser.add_argument('--path_dir_onnx', type=str, default='./ckpt/onnx/', help='ONNX path')
    parser.add_argument('--path_dir_tflite', type=str, default='./ckpt/tflite/', help='TFLIte path')
    args = parser.parse_args()
    return args


class LoadModel(object):
    @staticmethod
    def distill_internvideo2_small_patch14_224(_is_flash_attn: bool = False) -> distill_internvideo2_small_patch14_224:
        model = distill_internvideo2_small_patch14_224(
            clip_return_layer=1,
            use_flash_attn=_is_flash_attn,
            use_fused_rmsnorm=_is_flash_attn,
            use_fused_mlp=_is_flash_attn,
        )
        return model


if __name__=='__main__':
    args = get_args()
    quai_device = hub.Device(args.quai_device)
    path_onnx = path_.Path(os.path.join(args.path_dir_onnx, '{}.onnx'.format(args.model)))
    path_tflite = path_.Path(os.path.join(args.path_dir_tflite, '{}.tflite'.format(args.model)))

    if args.model == 'distill_internvideo2_small_patch14_224':
        opset_version = 13
        batch, channel, depth, height, width = 1, 3, 8, 224, 224
        dtype_input_video, dtype_input_mask = torch.float32, torch.int32
        input_names = ('input_video', 'input_mask')
        output_names = ('outputs_0', 'outputs_1')
        options_compile = '--target_runtime onnx'
        options_quantization = '--target_runtime tflite --truncate_64bit_tensors --quantize_io --quantize_io_type int8'

        input_video = torch.rand((batch, channel, depth, height, width), dtype=dtype_input_video)
        input_mask = torch.cat([
            torch.zeros(1, 1),
            torch.ones(1, 8 * int(16 * 16 * 0.75)),
            torch.zeros(1, 8 * int(16 * 16 * 0.25)),
        ], dim=-1).repeat(batch, 1).to(dtype_input_mask)

        spec_vid = (tuple(input_video.shape), str(dtype_input_video).replace('torch.', ''))
        spec_mask = (tuple(input_mask.shape), str(dtype_input_mask).replace('torch.', ''))
        input_specs = dict(
            input_video=spec_vid,
            input_mask=spec_mask,
        )
        inputs = dict(
            input_video=[np.random.randn(*spec_vid[0]).astype(quic_.QualcommAIHub.get_ndarr_type(spec_vid[1]))],
            input_mask=[np.random.randint(2, size=spec_mask[0]).astype(quic_.QualcommAIHub.get_ndarr_type(spec_mask[1]))]
        )
        calibration_data = inputs

        tensor_inputs = (input_video, input_mask)
        model = LoadModel.distill_internvideo2_small_patch14_224(_is_flash_attn=False)
        outputs = model(input_video, input_mask)
        printd('Output shapes: {}, {}'.format(outputs[0].shape, outputs[1].shape), _is_pause=False)
    else:
        raise NotImplementedError()

    # Calculate DNN computational complexity
    profiler_.DNNComplexity.fvcore(_model=model, _tensor_inputs=tensor_inputs, _max_depth=1)

    # Convert PyTorch to ONNX
    with torch.no_grad(), torch.cuda.amp.autocast():
        onnx_.ONNX.pytorch2onnx(
            _model=model,
            _tensor_inputs=tensor_inputs,
            _spath_onnx=path_onnx.str,
            _opset_version=opset_version,
            _input_names=input_names,
            _output_names=output_names,
            _is_simplify=True
        )
    printd('path_onnx: {}'.format(path_onnx.str), _is_pause=False)
    onnx_.ONNX.print_node_name(_spath_onnx=path_onnx.str)

    # Qualcomm AI Hub
    printd('[Qualcomm AI Hub: 1/6] Compiling', _is_pause=False)
    model_compiled = quic_.QualcommAIHub.compile(model=path_onnx.str, device=quai_device, input_specs=input_specs, options=options_compile)

    printd('[Qualcomm AI Hub: 2/6] Quantizing', _is_pause=False)
    model_quantized_compiled = quic_.QualcommAIHub.quantization(_model=model_compiled, _calibration_data=calibration_data, _weight_dtype=hub.QuantizeDtype.INT8, _activations_dtype=hub.QuantizeDtype.INT8, _device=quai_device, _options=options_quantization)
    model_compiled = model_quantized_compiled

    printd('[Qualcomm AI Hub: 3/6] Profiling', _is_pause=False)
    quic_.QualcommAIHub.profile(_target_model=model_compiled, _device=quai_device)

    printd('[Qualcomm AI Hub: 4/6] Inferencing', _is_pause=False)
    inference_job = quic_.QualcommAIHub.inference(_target_model=model_compiled, _device=quai_device, _inputs=inputs)

    printd('[Qualcomm AI Hub: 5/6] Downloading output data', _is_pause=False)
    output_data = quic_.QualcommAIHub.download_output(_inference_job=inference_job)

    printd('[Qualcomm AI Hub: 6/6] Downloading TFLite model', _is_pause=False)
    quic_.QualcommAIHub.download_target_model(_model_compiled=model_compiled, _spath_file=path_tflite.str)
