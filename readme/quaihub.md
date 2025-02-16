# Qualcomm AI Hub


## Table of contents
1. [Notice](#notice)
2. [Run](#run)


## 1. Notice <a name="notice"></a>
- I recommend that you should ignore the commented instructions with an octothorpe, #.


## 2. Run <a name="run"></a>
### 1. InternVideo2
```bash
(Container) $ model="distill_internvideo2_small_patch14_224"
(Container) $ quai_device="QCS8550 (Proxy)"
(Container) $ bash ./script/container/bash_run_quaihub.sh "${model}" "${quai_device}"
```


## 3. Error <a name="error"></a>
### 1. opset version for ONNX
You should check the opset version when converting PyTorch model to ONNX if you encounter below error related to ONNX.
```bash
[2025-02-16 11:46:25,460] [INFO] The branches of 'If' node '.clip_projector/cross_attn/If_1' produce incompatible shapes.
```