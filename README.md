# [ICLR 2025] SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes

This repository provides the code for **SSLAM**([Paper](https://openreview.net/forum?id=odU59TxdiB)), a novel self-supervised learning framework designed to enhance audio representation quality for both polyphonic and monophonic soundscapes.

---

## What, Why, and How

### Why
Real-world audio is often polyphonic—featuring multiple overlapping sound sources—yet most self-supervised audio models are trained and benchmarked on monophonic datasets (such as isolated environmental sounds or speech). This mismatch raises concerns about their robustness in practical applications, particularly in multi-modal systems like large language models. SSLAM was developed to bridge this gap by enhancing models’ abilities to learn from the complex, overlapping sounds encountered in everyday scenarios.

### What
**Self-Supervised Learning from Audio Mixtures (SSLAM)** is a novel approach in audio self-supervised learning. It is designed to improve performance on polyphonic audio data while still maintaining strong results on monophonic benchmarks. By incorporating mixtures of audio signals during training, SSLAM achieves new state-of-the-art results—demonstrating improvements of up to 3.9% on AudioSet-2M and up to 9.1% on polyphonic datasets.

### How
SSLAM employs a training strategy that utilizes polyphonic audio mixtures to learn richer, more generalizable representations. The approach involves:
- Training self-supervised audio models on mixtures rather than isolated sounds, incorporating a novel source retention loss that preserves the unique characteristics of each audio source. 
- Evaluating the models on both standard monophonic SSL benchmarks and high-quality polyphonic datasets.
- Demonstrating significant improvements in performance through both linear evaluation and fine-tuning regimes, with reported mean average precision (mAP) gains of up to 3.9% on AudioSet-2M and 9.1% on polyphonic data.

---

## **Inference Mode**

> **Note**: If you are already using [EAT](https://github.com/cwx-worst-one/EAT/tree/main) in your evaluation/inference pipeline, you can simply replace the weights with SSLAM weights, as the inference and evaluation code is identical to EAT.

If not, follow the steps below for installation:

```bash
conda create --prefix /path/to/sslam_eval_env -y python=3.9.13

/path/to/sslam_eval_env/bin/python -m pip install pip==24.0 # downgrade pip

/path/to/sslam_eval_env/bin/pip install -r SSLAM_Inference/requirements_sslam_eval.txt
```

---

#### **Model Weights**

We provide both pre-trained and AudioSet-2M fine-tuned model weights:

- [**Pre-Trained**](https://drive.google.com/drive/folders/1aA65-qQCHSCrkiDeLGUtn1PiEjJi5HS8?usp=sharing)
- [**AS2M Fine-Tuned**](https://drive.google.com/drive/folders/1Yy38IyksON5RJFNM7gzeQoAOSPnEIKp2?usp=sharing) (SOTA mAP of 50.2)

---

#### **Using SSLAM**

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


## Checklist 
- [x] Inference Mode
- [ ] Pre-Training

---

## **Acknowledgements**

Our code is primarily based on [EAT](https://github.com/cwx-worst-one/EAT/tree/main) and [data2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)  with additional concepts and components adapted from  [AudioMAE](https://github.com/facebookresearch/AudioMAE).


## Citation

If you find our work useful, please cite it as:  

```bibtex
@inproceedings{alex2025sslam,
  title={{SSLAM}: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes},
  author={Tony Alex and Sara Atito and Armin Mustafa and Muhammad Awais and Philip J B Jackson},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=odU59TxdiB}
}