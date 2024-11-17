# Evaluating Language Models for Open-Ended Tasks in Real Computer Environments

## Introduction
Language agents exhibit the capability to utilize natural language for a variety of complex tasks across different environments, particularly when built on large language models (LLMs). Notably, language model agents that can effectively execute intricate computer tasks introduces new research area for autonomous agents and also have the potential to revolutionize human-computer interaction. 

## Experiment
In this experiment, models and datasets from Hugging Face are utilized. The Llama-3.2-1B-Instruct model serves as the baseline, while both Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct are fine-tuned using the Mind2Web dataset (Deng et al., 2023). The GPU used for fine-tuning is the P100 available on Kaggle notebooks. Due to limited GPU resources, 8-bit quantization is applied during fine-tuning. The fine-tuned models are accessible at https://huggingface.co/ShinDC/llama_finetune_mind2web for the 3B model and https://huggingface.co/ShinDC/llama_finetune_mind2web_1B for the 1B model.

The evaluation employs the OSworld benchmark (Xie et al., 2024) to assess both baseline and fine-tuned models within computer environments. Due to time-limit error, a subset of 312 out of 369 tasks was used 

The evaluation metric, drawn from OSWorld (Xie et al., 2024), is based on the success rate for each domain and the overall average.

## Fine-tuning
To start fine-tuning, please login to huggingface-cli using your huggingface token, then run 
```
python finetune/finetune.py
```