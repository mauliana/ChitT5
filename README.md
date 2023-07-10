# FirstInteraction-FlanT5
Flan-T5 model fine-tuned in the First Interaction domain for Human-Robot Interaction purposes.

## Name
Flan-T5 based First Interaction for Human-Robot Interaction

## Description
This repository contain code to develop the conversation module on robot with first interaction theme. The text generation architecture for the conversation response is based on transformer seq2seq pretrained model [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5). The model is fine-tuned with conversation dataset using huggingface [trainer](https://huggingface.co/docs/transformers/main_classes/trainer) API with pytorch. Three dataset used for fine-tuning process:
- [DailyDialog](https://aclanthology.org/attachments/I17-1099.Datasets.zip) Datasets
- [CoQA](https://stanfordnlp.github.io/coqa/) datasets
- Small-talk dataset (a handcrafted datasets consist of the combination of english conversation for beginner which is available in some english learning website/blog)

## Usage
To run the first_interact_demo:
- Install all the python requirement modules for simulation.
- Download the required [model file](https://seafile.rlp.net/d/8da927ae0b244ece9c39/) and store it in model folder.
- Re-check whether the path to the model folder correct or not in first_interact_demo.py before running the code.

To reproduce the model:
- Install all the python requirement modules for training.
- Download the datasets and code on each folder.
- Run data_preprocessing.py
- Run training.py
- Then, export the best_model files from training result to onnx format using t5-exporting-to-onnx

## Installation Requirement
Simulation:

Reproduce the model:

## References
<a id="1">[1]</a> 
Chung, Hyung Won, et al. "Scaling instruction-finetuned language models." arXiv preprint arXiv:2210.11416 (2022).
