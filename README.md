# Token-Level Self-Play with Importance-Aware Guidance for Large Language Models
<!-- 
<div align="center">
[![arXiv](https://img.shields.io/badge/arXiv-2401.01335-b31b1b.svg?style=flat)](https://arxiv.org/abs/2401.01335)  
[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg?style=flat)](https://proceedings.mlr.press/v202)  
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)  
[![HuggingFace Datasets](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/Ultrachat200k)  
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/Qwen/Qwen1.5-1.8B)
</div> -->

## 🔍 Overview
SWIFT (Self-Play Weighted Fine-Tuning) is a simple yet effective method for aligning large language models via token-level importance weighting.



## 🔧 Installation

1. **Create and activate the conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate SWIFT
   pip install -r requirements.txt
   ```

## 📊 Data Preparation

Download the Ultrachat200k dataset for SFT [Link SFT data](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter1) and DPO from [Link DPO data](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) and organize them under `data/`:

Ensure the following directories exist:

```
data/
├── Ultrachat200k/
│   ├── DPO/
│   ├── SFT/
│   ├── SPIN/
│   └── SWIFT/
```

## 🤖 Model Preparation

We use Qwen1.5-1.8B and GPT2-1.5B as the base LLM and zephyr-7b-sft-full as teacher LLM.

<!-- ```bash
huggingface-cli download --resume-download Qwen/Qwen1.5-1.8B --local-dir models/Qwen1.5-1.8B
``` -->

## 🚀 Training Pipeline

### Part 0: Download Base Model Checkpoint

Download any provided starting checkpoint:

```bash
python scripts/download.py
```

---

### Part 1: SWIFT for Qwen1.5-1.8B

```bash
bash scripts/_SPIN_full.sh
bash scripts/_SWIFT_full.sh
```

## Citation

```bibtex
@inproceedings{letoken,
  title={Token-Level Self-Play with Importance-Aware Guidance for Large Language Models},
  author={Le, Tue and Vuong, Hoang Tran and Tran, Quyen and Van, Linh Ngo and Harandi, Mehrtash and Le, Trung},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

