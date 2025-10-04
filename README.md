# 🔊 [ICLR 2025] SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes

[![Conference Paper](https://img.shields.io/badge/ICLR-2025-blue)](https://openreview.net/forum?id=odU59TxdiB)

🚀 **SSLAM** is a self-supervised learning framework designed to enhance audio representation quality for both **polyphonic(multiple overlapping sounds)** and monophonic soundscapes. Unlike traditional SSL models that focus on monophonic data, SSLAM introduces a novel **source retention loss** and **audio mixture training**, significantly improving performance on real-world polyphonic audio.

🔗 **[Paper](https://openreview.net/pdf?id=odU59TxdiB) | [ICLR 2025 Poster: Video & Slides](https://iclr.cc/virtual/2025/poster/28347)  | [Open Review](https://openreview.net/forum?id=odU59TxdiB) | [🤗 Models](https://huggingface.co/ta012/SSLAM_pretrain) | [Models(Google Drive)](https://drive.google.com/drive/folders/1G0icv-hdqDEqnfP4EFszMXhFnWWM09gT?usp=sharing)**


---

# 📋 Table of Contents
- [Why SSLAM?](#why-sslam)
- [Key Features](#key-features)
- [Results](#results)
- [Inference Mode](#️inference-mode)
  - [Hugging Face Inference](#hugging-face-inference)
  - [Inference Installation](#minimal-installation-for-inference/evaluation)
  - [Model Weights](#model-weights)
  - [Using SSLAM](#using-sslam)
- [Training Mode](#training-mode)
  - [Training Installation](#training-installation)
  - [Data Preparation](#️data-preparation)
  - [Pre-Training](#pre-training)
- [Checklist](#checklist)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## 🔍Why SSLAM? 
🔊 **Real-world audio is polyphonic**—multiple overlapping sound sources are common in everyday environments.  
❌ **Existing SSL models focus on monophonic audio,** limiting their ability to generalize to real-world scenarios. Their benchmarks are primarily monophonic, and their pre-training does not account for polyphonic environments.   
💡 **SSLAM bridges this gap** by introducing **self-supervised learning from audio mixtures**, enabling robust learning across **both monophonic and polyphonic soundscapes**.

---

## 🎼Key Features
✅ **Self-Supervised Learning from Audio Mixtures (SSLAM)** – improving robustness to real-world polyphonic audio  (multiple overlapping sounds).  
✅ **Source Retention Loss** – ensures the integrity of each sound source even in complex mixtures.  
✅ **SOTA Performance** – Achieves **+3.9% mAP improvement** on AudioSet-2M and **+9.1% on polyphonic datasets**.  

---
## 📊Results

### 1. Standard Audio-SSL Benchmark Datasets
![Standard Audio-SSL Benchmark](assets/as2m_results.png)

### 2. Polyphonic Datasets
![Polyphonic Datasets](assets/poly_results.png)

---

## **🔍️Inference Mode**

## 🤗Hugging Face Inference

You can run SSLAM for feature extraction without cloning this repo via [Hugging Face](https://huggingface.co/ta012/SSLAM_pretrain).

- **[Pretrained](https://huggingface.co/ta012/SSLAM_pretrain) AudioSet 2M - ViT Base - Feature Extraction:**
- **[Finetuned on AudioSet 2M(50.2 mAP)](https://huggingface.co/ta012/SSLAM_AS2M_Finetuned) - Feature Extraction and Event Labels:**


## 📥Minimal Installation for Inference/Evaluation 
> **Note**: If you are already using [EAT](https://github.com/cwx-worst-one/EAT/tree/main) in your evaluation/inference pipeline, you can simply replace the weights with SSLAM weights, as the inference and evaluation code is identical to EAT.

If not, follow the steps below for installation:

To simplify installation and avoid dependency conflicts, we've included a cloned copy of fairseq (`SSLAM_Inference/cloned_fairseq_copy/fairseq`) in the repository instead of requiring a direct fairseq installation.


```bash
conda create --prefix /path/to/sslam_infer_minimal_env -y python=3.9.13

/path/to/sslam_infer_minimal_env/bin/pip install -r SSLAM_Inference/requirements_sslam_infer_minimal.txt
```

**Important**: Update the fairseq path in these files:

- `SSLAM_Inference/evaluation/eval.py`
- `SSLAM_Inference/feature_extract/feature_extract.py`
- `SSLAM_Inference/inference/inference.py`

Look for the `fairseq_path` variable and update it to point to the included clone:
```python
fairseq_path = '/absolute/path/to/SSLAM/SSLAM_Inference/cloned_fairseq_copy/fairseq/'
```


---

## 📦Model Weights

| Model Type               | Link                                                                                       |
|--------------------------|--------------------------------------------------------------------------------------------|
| **Pre-Trained**          | [Download](https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8?usp=sharing) |
| **AS2M Fine-Tuned** (50.2 mAP) | [Download](https://drive.google.com/drive/folders/1Yy38IyksON5RJFNM7gzeQoAOSPnEIKp2?usp=sharing) |
---

#### 🚀**Using SSLAM**

We provide scripts to use SSLAM in the following ways:

##### 1. **Audio Feature (Representation) Extraction Using SSLAM Encoder**

```bash
cd SSLAM_Inference/scripts
bash feature_extract.sh 
```

##### 2. **Inference on Single Audio WAV File**

```bash
cd SSLAM_Inference/scripts
bash inference.sh
```

##### 3. **Evaluation on AudioSet-2M Evaluation Set**

```bash
cd SSLAM_Inference/scripts
bash evaluate_AS2M_finetuned.sh # Reported mAP: 50.2
```

---
## **📈Training Mode**
We cover the self-supervised pre-training, fine-tuning and linear evaluation under this section.

#### **📥Training Installation** 

For training its better to install the fairseq in editable mode,

```bash
conda create --prefix /path/to/sslam_env -y python=3.9.13 ## env used for training
/path/to/sslam_env/bin/python -m pip install pip==24.0 # downgrade pip
cd SSLAM/
git clone https://github.com/facebookresearch/fairseq.git

##IMPORTANT: Copy the Pre-Training/SSLAM_Stage2 directory to SSLAM/fairseq 
## so that the resultant path is SSLAM/fairseq/SSLAM_Stage2/.
cd fairseq/

## install all requirements apart from fairseq
/path/to/sslam_env/bin/pip install -r SSLAM_Stage2/requirements_sslam.txt
## install fairseq in editable mode
/path/to/sslam_env/bin/pip install --editable ./
```
#### 🗄️Data Preparation
We utilised AudioSet-2M (full set) for pre-training. For this phase, only the `train.tsv` file is required. Refer to [train.tsv for AudioSet-20K](data_manifests/manifest_as20k/train.tsv) to prepare the train.tsv file for your downloaded copy of AudioSet-2M.

#### 🚀Pre-Training

**Note:** This repository focuses solely on Stage 2 pre-training, which introduces our novel SSLAM pre-training strategy. 

To begin Stage 2, you'll need a Stage 1 checkpoint. In our complete pre-training process, Stage 1 mirrors the approach in [EAT](https://github.com/cwx-worst-one/EAT/tree/main) and achieves similar performance. For convenience, we use the EAT checkpoint as the Stage 1 checkpoint.

Download the [EAT](https://github.com/cwx-worst-one/EAT/tree/main) epoch 10 checkpoint using the link provided by the [EAT](https://github.com/cwx-worst-one/EAT/tree/main) repository: [EAT-base_epoch10_pt.pt](https://drive.google.com/file/d/10pklbY_fKraQUIBizSg1kv4lJXNWxpxl/view?usp=sharing).

*Only the contents of the **models/** folder and a few parameters in the pre-training script differ between Stage 1 and Stage 2.*

```bash
cd SSLAM/fairseq/SSLAM_Stage2/scripts/
bash pretrain_stage2.sh
```


## 📌Checklist 
- [x] Inference Mode
- [x] Pre-Training
- [x] Finetuning & Linear Evaluation


---

## 🙏Acknowledgements

Our code is primarily based on [EAT](https://github.com/cwx-worst-one/EAT/tree/main) and [data2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)  with additional concepts and components adapted from  [AudioMAE](https://github.com/facebookresearch/AudioMAE).


## 📜Citation

If you find our work useful, please cite it as:  

```bibtex
@inproceedings{alex2025sslam,
  title={{SSLAM}: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes},
  author={Tony Alex and Sara Atito and Armin Mustafa and Muhammad Awais and Philip J B Jackson},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=odU59TxdiB}
}
