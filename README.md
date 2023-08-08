[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LW9jbkQZ)
# Final Project

This is a **group assignment**.

## Code Implementation & Technical Report

The final deliverables include a 4-page IEEE-format report, code implementation and a detailed GitHub readme file.

The final project is due Wednesday, April 26 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/469791/assignments/5497341) in the Canvas assignment.

## Training Data

The training data set is the same for every team in this course.

You can download the training data from the Canvas page:

* ["data_train.npy"](https://ufl.instructure.com/files/76267874/download?download_frd=1)
* ["labels_train.npy"](https://ufl.instructure.com/files/76267876/download?download_frd=1)


## Getting Started

### Dependencies

1. Create a new conda environment `ml_experts` with the following dependencies.

`conda create -n ml_experts python=3.7 torchvision cudatoolkit=11.3 pytorch=1.12.1=gpu_cuda* -c pytorch`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge imutils`

`conda install -c conda-forge tqdm`

`conda install -c anaconda scikit-learn`

### Installation

1. Clone the repo.

`git clone https://github.com/UF-FundMachineLearning-Sp23/final-project-ml-experts.git`

2. Setup and activate environment

`conda activate ml_experts`

## Usage

### Training
1. Put `data_train.npy` and `labels_train.npy` in the base directory. The shapes should be (270000, n_samples) and (n_samples, ) respetively. 
2. `python prepare_train_data.py`
3. `python train.py`

How it works:
1. `prepare_train_data.py` creates images from `data_train.npy` and save them in `logo_photos` folder. Next it creates `dataset/train/` and `dataset/val/` folders and put the images according to `VAL_SPLIT` ratio.
2. `train.py` imports pretrained `ResNet-50` model and finetune it on provided train and validation data. The model is saved at `output/finetune_model.pth`.

Note: If you want to change any parameters of training, please do it in `utils/config.py`. Here are some variables that you can change.
1. `TRAIN_DATA` : Name of your data file. (a .npy file)
2. `TRAIN_LABELS` : Name of your labels file. (a .npy file)
3. `VAL_SPLIT` : Train/Validation ratio. (default is 70/30)
4. `IMAGE_SIZE` : Input image size to model. (default is 224, must be multiple of 32)
5. `FINETUNE_BATCH_SIZE` : Training batch size. (default is 64)
6. `EPOCH` : Training epoch. (default is 100)
7. `LR` : Learning rate. (default is 0.001)
8. `FINETUNE_MODEL` : Directory and name for saving the model

### Testing
1. Put `data_test.npy` and `labels_test.npy` in the base directory. The shapes should be (270000, n_samples) and (n_samples, ) respetively. 
2. `python test.py`

How it works:
1. The `test.py` creates images from `data_test.npy` and saves them in `dataset/test/` folder. Next it generates `pred_list` that contains predicted labels. It also prints a classification report.
2. If you trained and saved a model with different name then use: `python test.py --path <path to your model>`

## Authors
1. Imran Nasrullah [email](nasrullah.i@ufl.edu)
2. Nicholai Platonoff [email](nplatonoff@ufl.edu)
3. Adnan Abdullah [email](adnanabdullah@ufl.edu)
