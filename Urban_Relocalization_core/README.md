# Re-localization in urban settings

## The folder structure

| Name of Folder   | Contains                     |
| ---------------- | ---------------------------- |
| data             | folder for all the datasets  |
| data/dataset_XXX | individual datasets          |
| models_          | weights of the latest model  |
| summary_         | training and testing results |

## Prerequisites

- Pip
- Python 3
- Virtual Environment

## Install instructions

```
open terminal
$ git clone https://github.com/Shivam-Bhardwaj/Urban_relocalization.git
$ cd Urban_relocalization
$ virtualenv --no-site-packages -p python3 venv 
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Training instructions

```
open terminal
$ source venv/bin/activate
$ python train.py --image_path data/dataset_XXX --metadata_path data/dataset_XXX/train.txt
```

The results will be stored in models_ folder in the root directory

## Batch Test instructions

```
open terminal
$ source venv/bin/activate
$ python single_test.py --image_path data/dataset_XXX/ --metadata_path data/dataset_XXX/test.txt --weights_path models_/XXX_net.pth 

```

A file named `result.txt` will be created in the root folder.

To visualize the results, run `plot_compare.py` 

`NOTE: replace result_file and  ground_truth with the test results and ground truth respectively`

The code was tested on the following specifications

- **CPU:** `Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz`
- **GPU:** `Nvidia GeForce GTX 1080 Ti Mobile`
- **OS:** `Ubuntu 16.04.6 LTS (Xenial Xerus)`
- **Kernal:** `4.15.0-48-generic`



