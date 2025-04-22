# Weighted Self-Play Fine‑Tuning (WSPIN)
<!-- 
<div align="center">
[![arXiv](https://img.shields.io/badge/arXiv-2401.01335-b31b1b.svg?style=flat)](https://arxiv.org/abs/2401.01335)  
[![ICML](https://img.shields.io/badge/ICML-2025-blue.svg?style=flat)](https://proceedings.mlr.press/v202)  
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)  
[![HuggingFace Datasets](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/Ultrachat200k)  
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/Qwen/Qwen1.5-1.8B)
</div> -->

## 📌 Table of Contents
- [Overview](#🔍-overview)
- [Installation](#🔧-installation)
- [Data Preparation](#📊-data-preparation)
- [Model Preparation](#🤖-model-preparation)
- [Training Pipeline](#🚀-training-pipeline)
  - [Part 0: Download Checkpoint](#part-0-download-checkpoint)
  - [Part 1: Base DPO on Qwen1.5-1.8B](#part-1-base-dpo-on-qwen15-18b)
  - [Part 2: Iteration 0 (ITE0)](#part-2-iteration-0-ite0)
  - [Part 3: Iteration 1 (ITE1)](#part-3-iteration-1-ite1)
  - [Part 4: Iteration 2 (ITE2)](#part-4-iteration-2-ite2)
  - [Part 5: Iteration 3 (ITE3)](#part-5-iteration-3-ite3)
- [Repository Structure](#📝-repository-structure)
- [Acknowledgements](#🙏-acknowledgements)
- [Citation](#📄-citation)
- [License](#📜-license)

---

## 🔍 Overview

Weighted Self‑Play Fine‑Tuning (WSPIN) builds on the Self-Play Fine‑Tuning (SPIN) framework by incorporating token-level importance weights into each iteration. While SPIN refines the model by distinguishing human demonstrations from its own generated responses, WSPIN further prioritizes high-value tokens—estimated via a teacher–student distillation approach—during optimization. Experimental results demonstrate notable improvements over vanilla SPIN across standard LLM benchmarks.

## 🔧 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-org/WSPIN.git
   cd WSPIN
   ```

2. **Create and activate the conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate wspin
   ```

3. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Preparation

Download the Ultrachat200k dataset from Hugging Face and organize under `data/`:

```bash
huggingface-cli download --resume-download Ultrachat200k/DPO --local-dir data/Ultrachat200k/DPO --repo-type dataset
git lfs pull --include "Ultrachat200k/SPIN/*"
```

Ensure the following directories exist:

```
data/
├── Ultrachat200k/
│   ├── DPO/
│   ├── SPIN/
│   └── WSPIN/
```

## 🤖 Model Preparation

We use Qwen1.5-1.8B as the base LLM.

```bash
huggingface-cli download --resume-download Qwen/Qwen1.5-1.8B --local-dir models/Qwen1.5-1.8B
```

## 🚀 Training Pipeline

### Part 0: Download Checkpoint

Download any provided starting checkpoint:

```bash
python scripts/download.py
```

---

### Part 1: Base DPO on Qwen1.5-1.8B

Train the first DPO model on Ultrachat200k/DPO:

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=models/Qwen1.5-1.8B/SFT/ \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/DPO"]'
```

---

### Part 2: Iteration 0 (ITE0)

#### Step 2.1: Generate Samples for ITE0

```bash
bash scripts/ite0_generate.sh
```

#### Step 2.2: Compute Token Importance Weights

```bash
bash scripts/ite0_weight_estimation.sh
```

#### Step 2.3: Train SPIN (Vanilla) for ITE0

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=models/Qwen1.5-1.8B/SFT/ \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/SPIN/ite0"]'
```

#### Step 2.4: Train WSPIN for ITE0

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=models/Qwen1.5-1.8B/SFT/ \
  loss=tis-dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/WSPIN/ite0"]'
```

---

### Part 3: Iteration 1 (ITE1)

#### Step 3.1: Generate Samples for ITE1

```bash
bash scripts/ite1_generate_SPIN.sh
bash scripts/ite1_generate_WSPIN.sh
```

#### Step 3.2: Compute Token Importance Weights

```bash
bash scripts/ite1_weight_estimation.sh
```

#### Step 3.3: Train SPIN for ITE1

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite0 \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/SPIN/ite0","Ultrachat200k/SPIN/ite1"]'
```

#### Step 3.4: Train WSPIN for ITE1

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite0 \
  loss=tis-dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/WSPIN/ite0","Ultrachat200k/WSPIN/ite1"]'
```

---

### Part 4: Iteration 2 (ITE2)

#### Step 4.1: Generate Samples for ITE2

```bash
bash scripts/ite2_generate_SPIN.sh
bash scripts/ite2_generate_WSPIN.sh
```

#### Step 4.2: Compute Token Importance Weights

```bash
bash scripts/ite2_weight_estimation.sh
```

#### Step 4.3: Train SPIN for ITE2

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite1 \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/SPIN/ite1","Ultrachat200k/SPIN/ite2"]'
```

#### Step 4.4: Train WSPIN for ITE2

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite1 \
  loss=tis-dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/WSPIN/ite1","Ultrachat200k/WSPIN/ite2"]'
```

---

### Part 5: Iteration 3 (ITE3)

#### Step 5.1: Generate Samples for ITE3

```bash
bash scripts/ite3_generate_SPIN.sh
bash scripts/ite3_generate_WSPIN.sh
```

#### Step 5.2: Compute Token Importance Weights

```bash
bash scripts/ite3_weight_estimation.sh
```

#### Step 5.3: Train SPIN for ITE3

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SPIN/ite2 \
  loss=dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/SPIN/ite2","Ultrachat200k/SPIN/ite3"]'
```

#### Step 5.4: Train WSPIN for ITE3

```bash
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite2 \
  loss=tis-dpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/WSPIN/ite2","Ultrachat200k/WSPIN/ite3"]'
```

---

## 📝 Repository Structure

```
.
├── README.md                 # This documentation
├── environment.yml           # Conda environment spec
├── requirements.txt          # Python dependencies
├── scripts/                  # Utility scripts
│   ├── download.py           # Checkpoint downloader
│   ├── ite0_generate.sh      # Generate ITE0 samples
│   ├── ite*_weight_estimation.sh # Token weight estimators
│   └── ite*_generate_SPIN.sh
├── train.py                  # Core training script
├── utils.py                  # Helper functions
├── preference_datasets.py    # Data loaders and transforms
├── trainers.py               # SPIN and WSPIN implementations
├── transform_config.py       # Weight transformation methods
├── data/                     # Dataset directory
│   └── Ultrachat200k/
│       ├── DPO/
│       ├── SPIN/
│       └── WSPIN/
└── output/                   # Model checkpoints and logs
```

## 🙏 Acknowledgements

Thanks to the authors of SPIN and TIS-DPO for foundational ideas:
- **SPIN**: Zixiang Chen et al., *Self-Play Fine‑Tuning Converts Weak Language Models to Strong Language Models*, ICML 2024.
- **TIS-DPO**: Aiwei Liu et al., *Token‑level Importance Sampling for Direct Preference Optimization*, ICLR 2025.

## 📄 Citation

If you find this work useful, please cite:

<!-- ```bibtex
@inproceedings{your2025wspin,
  title={{WSPIN}: Weighted Self-Play Fine-Tuning with Token Importance},
  author={Your Name and Collaborators},
  booktitle={Proceedings of ICML 2025},
  year={2025},
  url={https://github.com/your-org/WSPIN}
}
``` -->

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

