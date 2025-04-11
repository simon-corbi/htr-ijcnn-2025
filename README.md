# htr-ijcnn-2025

Official code for the paper "" presented in IJCNN 2025.

Authors: Simon Corbillé, Elisa H. Barney Smith

Machine learning team from Luleå University of Technology

## Installation
Code test with:
* Python 3.11
* albumentations 1.4.15
* albucore 0.016

```
pip install -r requirements.txt
```

For GPU: install Pytorch for GPU

## Data
### Format dataset

|              | Download link | Python file format |
| -------------| -------------| ------------- |
| IAM          | [Link](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | src/data/format/format_iam.py  |
| Cipher T1    | [Link](https://rrc.cvc.uab.es/?ch=27&com=downloads) 		  | src/data/format/format_ciphers.py  |
| Cipher T2A   | [Link](https://rrc.cvc.uab.es/?ch=27&com=downloads)  		  | src/data/format/format_ciphers.py  |
| Cipher T2B   | [Link](https://rrc.cvc.uab.es/?ch=27&com=downloads)  		  | src/data/format/format_ciphers.py  |
| Cipher T3A   | [Link](https://rrc.cvc.uab.es/?ch=27&com=downloads) 		  | src/data/format/format_ciphers.py  |
| Cipher T3B   | [Link](https://rrc.cvc.uab.es/?ch=27&com=downloads) 		  | src/data/format/format_ciphers.py  |
| NorHand v1   | [Link](https://zenodo.org/records/6542056) 		  | Already formatted  |

### Images size

for parameters: height_max and width_max

|              | Image height | Image width |
| -------------| -------------| ------------- |
| IAM          | 128          | 1700  |
| Cipher T1    | 120 		  | 1900  |
| Cipher T2A   | 96  		  | 768   |
| Cipher T2B   | 190  		  | 2100  |
| Cipher T3A   | 84 		  | 1120  |
| Cipher T3B   | 190 		  | 2256  |
| NorHand v1   | 200  		  | 2256  |

### Example config file (.json)

cf. directory configuration

### Text format

IAM, Norhand: text

	--read_txt_format "RAW" 
	--filter_txt "NO" 
	--compute_wer 1 
	--use_wer_formula_for_cer 0 
	--space_value "RAW" 
	
Cipher DBs: class labels are separated by space character

	--read_txt_format "CLASSES_SPACED_WITH_SPACE" 
	--filter_txt "CLEAR_TEXT" 
	--compute_wer 0 
	--use_wer_formula_for_cer 1 
	--space_value "TEXT" 
	


## Train

src/train/train_crnn.py

Need 2 parameters:
* configuration file cf. example
* log directory

example:
```
python src/train/train_crnn.py configuration/config_cpu_demo_iam.json logs
```


## Evaluate
src/evaluate/evaluate_crnn.py

parameters
config file like for training
path model pretrained IAM

link file

Performance IAM line level
Validation (976 lines)
CER: 2.58% WER: 10.80% 

Test(2915 lines)
Main : CER: 3.85% WER: 14.57% 

## Reference
CRNN from "Best Practices for a Handwritten Text Recognition System"

git: https://github.com/georgeretsi/HTR-best-practices/

article: https://arxiv.org/abs/2404.11339

## Citation
If you find this work useful, please consider citing:
```
@inproceedings{corbille2025,
  title={to do},
  author={to do},
  booktitle={to do},
  pages={to do},
  year={2025},
  organization={to do}
}
```
