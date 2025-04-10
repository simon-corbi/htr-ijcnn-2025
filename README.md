# htr-ijcnn-2025

Official code for the paper "" presented in IJCNN 2025.

Authors: Simon Corbillé, Elisa H. Barney Smith

Machine learning team from Luleå University of Technology

## Installation
Code test with Python 3.11

albumentations 1.4.15
albucore 0.016

```
pip install -r requirements.txt
```

For GPU used install Pytorch for GPU

## Data
### Format dataset

### Hyperparameter
height, width image

### Example config file

.json

### Text format

## Train

src/train/train_crnn.py

need parameters
configuration file cf. example

log directory


## Evaluate
model pretrained IAM

link file

perf on val / test

## Reference
CRNN from "Best Practices for a Handwritten Text Recognition System"

git: https://github.com/georgeretsi/HTR-best-practices/

article: https://arxiv.org/abs/2404.11339

## Citation
If you find this work useful, please consider citing:
```
@inproceedings{retsinas2022best,
  title={Best practices for a handwritten text recognition system},
  author={Retsinas, George and Sfikas, Giorgos and Gatos, Basilis and Nikou, Christophoros},
  booktitle={International Workshop on Document Analysis Systems},
  pages={247--259},
  year={2022},
  organization={Springer}
}
```
