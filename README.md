

# SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes



This repository provides the code for **SSLAM**, a novel self-supervised learning framework designed to enhance audio representation quality for both polyphonic and monophonic soundscapes.

---

## **Inference Mode**

> **Note**: If you are already using EAT in your evaluation/inference pipeline, you can simply replace the weights with SSLAM weights, as the inference and evaluation code is identical to EAT.

If not, follow the steps below for installation:

```bash
# Create a new conda environment
conda create --prefix /path/to/sslam_eval_env -y python=3.9.13

# Downgrade pip if necessary
/path/to/sslam_eval_env/bin/python -m pip install pip==24.0

# Install required dependencies
/path/to/sslam_eval_env/bin/pip install -r SSLAM_Inference/requirements_sslam_eval.txt
```

---

### **Model Weights**

We provide both pre-trained and AudioSet-2M fine-tuned model weights:

- [**Pre-Trained**](https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8?usp=sharing)
- [**AS2M Fine-Tuned**](https://drive.google.com/drive/folders/1Yy38IyksON5RJFNM7gzeQoAOSPnEIKp2?usp=sharing) (SOTA mAP of 51.2)

---

### **Using SSLAM**

We provide scripts to use SSLAM in the following ways:

#### 1. **Audio Feature (Representation) Extraction Using SSLAM Encoder**

```bash
cd SSLAM_Inference/scripts

# Update paths in the script before running
bash feature_extract.sh
```

#### 2. **Inference on Single Audio WAV File**

```bash
cd SSLAM_Inference/scripts

# Update paths in the script before running
bash inference.sh
```

#### 3. **Evaluation on AudioSet-2M Evaluation Set**

```bash
# Update the evaluation paths in data_manifests/manifest_as20k/eval.tsv
cd SSLAM_Inference/scripts

# Update paths in the script before running
bash evaluate_AS2M_finetuned.sh

# Reported mAP: 51.2
```

---


## Checklist 
- [x] Inference Mode
- [ ] Pre-Training

---

## **Acknowledgements**

Our code is primarily based on EAT, with additional concepts and components adapted from data2vec 2.0 and AudioMAE.


## **Citation**