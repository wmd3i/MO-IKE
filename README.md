# MO-IKE

This repository contains the code for our paper "Towards Reliable, Generalizable, and Specific In-Context Knowledge Editing via Multi-Objective Reinforcement Learning". The code is organized into several directories, each containing specific components of our framework.

## Directory Structure

- **utils/** - Utility functions for data processing, in-context learning construction, and frozen LLM API interactions
- **eval.py** - Evaluation code for assessing the performance of the knowledge editing
- **train.py** - Training code for the MO-IKE framework
- **Datasets/** - Contains the counterfact.json file used for training and evaluation. You may evaluate the performance of MO-IKE on other datasets by replacing the counterfact.json file with your desired dataset
- **requirements.txt** - Required Python packages for running the code

## Usage

### Installation

To install the required packages, run the following command:

```bash
git clone https://github.com/xuzhongwm/MO-IKE.git
cd MO-IKE
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Evaluation

```bash
python eval.py
```
