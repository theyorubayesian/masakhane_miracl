# masakhane_miracl

This repository guides our submission to the [WSDM 2023 Cup](https://www.wsdm-conference.org/2023/program/wsdm-cup): [MIRACL](https://project-miracl.github.io/)

## Setup & Installation

* Create an environment using either Conda or Venv

```bash
conda create -n miracl python=3.8 openjdk=11
conda activate miracl
```

* Clone the repo

```bash
git clone --recurse-submodules https://github.com/theyorubayesian/masakhane_miracl.git 
```

* Install `Pytorch>=1.10` suitable for your CUDA version. See [Pytorch](https://pytorch.org/get-started/previous-versions/#v1101)

* Install other requirements

```bash
pip install -r requirements.txt
```

* Login to [Weights & Biases](https://wandb.ai/masakhane-miracl/masakhane-miracl) where we are logging our experiments.

```bash
wandb login
```

* Hack away 🔨🔨

## Experiments

1. [Training on MS Marco & Reporting Zero-Shot Results on Mr.TyDi Swahili](docs/msmarco_finetuning_experiment.md)
2. [Zero-Shot Evaluation of the Dense Retriever on Miracl Dev Set](docs/evaluating_on_miracl_dev_set.md)
3. [Finetuning the Dense Retriever on the Miracl Train Set & Generating Rankings for the `testA` set](docs/miracl_finetuning_experiment.md)