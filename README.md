# Fine-Tuning Open-Source Small Language Models (SLMs) Using Unsloth on Google Colab

This repository contains the full notebook and Python script used to fine-tune an open-source Small Language Model (SLM) using the **Unsloth** optimization framework.  
The tutorial demonstrates how to prepare a custom dataset, apply LoRA adapters, export safetensors, and run inference — all within the memory limits of Google Colab.

---

## 📌 Contents

- `finetune_slm_unsloth.ipynb` — The full walkthrough notebook
- `finetune_slm_unsloth.py` — End-to-end training script
- Sample dataset format (`data.json`)
- Instructions for loading both pretrained and **locally downloaded safetensors** models
- Exporting LoRA adapters and converting them into `.safetensors`

---

## 📖 Medium Article (Part 1)

The full write-up, including explanations of preprocessing strategies, LoRA modules, hyperparameters, and inference testing, is available here:

👉 **Medium Article — Part 1:**  
[Click to see the Guide on Medium ](https://medium.com/@sayantanenator/fine-tuning-an-open-source-small-language-model-with-unsloth-on-google-colab-8d5316139723)

---

## 🚀 Part 2 Coming Soon

Part 2 will cover **production-grade deployment**, including:

- Training and hosting with **AWS SageMaker Training Jobs**
- Exposing models through **MLflow**
- Deploying to **HuggingFace Inference Endpoints**
- Lightweight deployments on **Azure Foundry**
- Best practices for running SLMs in resource-constrained environments

Stay tuned.

---

## 📜 License

This project is under the MIT License.
