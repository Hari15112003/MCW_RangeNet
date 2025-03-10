# RangeNet Configuration

## Overview
This project is configured for **semantic segmentation** using **RangeNet** with the **PyTorch** framework. The model is optimized using post-training quantization (PTQ) and quantization-aware training (QAT). The dataset used is **KITTI**, and inference is performed on CUDA-enabled devices.

## Configuration Details

### General Settings
- **Model Name:** rangenet
- **Framework:** PyTorch
- **Task:** Semantic Segmentation
- **CUDA Device:** 1
- **Input Shape:** [1, 5, 64, 2048]
- **Dataset:** KITTI
- **Pretrained Model:** artifacts/squeezeseg

### Quantization Settings
- **Auto Quantization:** False
- **Quantization Enabled:** True
- **Quantization Configuration:**
  - **Type:** rangenet_w8a8
  - **Parameter Bit-width:** 8
  - **Output Bit-width:** 8
  - **Input Quantization:** True
  - **Quantization Scheme:** tf_enhanced
  - **Techniques Used:** CLE (Cross Layer Equalization), BN (Batch Normalization)
- **Allowed Accuracy Drop:** 1%

### Paths
- **Dataset Path:** `/media/ava/DATA/aleesha/datasets`
- **Export Directory:** `/media/ava/DATA3/DATA/athirooban/shabari/harish/rangenet/artifacts`
- **Exported Model Name:** `rangenet_manual_ptq_5x64x2048`
- **Result Path:** `src/prediction`

### Quantization-Aware Training (QAT)
- **QAT Enabled:** True
- **QAT Model Name:** `rangenet_qat_5x64x2048`

## Usage Instructions
- Run by using  `quant.py --config path_to_config_file` 

Results will be stored in `src/prediction`.

## Requirements
- PyTorch
- CUDA Toolkit
- AIMET (AI Model Efficiency Toolkit) for quantization
- KITTI dataset (download and place in the dataset path)

## Notes
- Ensure that the CUDA device is available before running the model.
- If using QAT, make sure the training pipeline supports quantization-aware operations.
- The allowed accuracy drop for quantization is set to 1%; adjust as necessary for your requirements.

## References
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [AIMET Quantization](https://github.com/quic/aimet)

