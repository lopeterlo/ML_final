# ML_final

## Machine learning fall 2019 final project -  Dialogue Modeling Problem
Kaggle link: https://www.kaggle.com/c/ml2019fall-final-dialogue/overview


### Task
![image](https://github.com/lopeterlo/ML_final/blob/master/pic/Task_description.png)
### Data format
![image](https://github.com/lopeterlo/ML_final/blob/master/pic/data_format.png)
### Output format
![image](https://github.com/lopeterlo/ML_final/blob/master/pic/output_format.png)

## Prerequisites
* Python 3.6
* wget

## Installation
1. Clone the repo
2. Install packages through pip
```
pip install -r requirements.txt
```
## Usage
### Download dataset & Data preprocessing:
```
./download_dataset.sh
```
Three files (**train.json, valid.json, test.json**) would be downloaded in **./data** folder.

```
python3 data_preprocessing.py
```
This will generate structured data in **./struct_data** folder.
### Directly download preprocessed data:
```
./download_preprocessed_dataset.sh
```
This will directly download preprocessed data with last sentence length 300, false number 3

### Fine-tuning with Bert pretrained model:

```
cd BERT
bash ./train.sh [train_data] [valid_data] [valid_train]
```
### Or Download fine tuned Bert model:
```
bash ./download_fine_tune_BERT.sh
```

### Testing:
```
cd BERT
bash ./test.sh [test_data] [model_path] [output]
```

### Config file:
If you want to adjust hyperparameter or change cuda device id, please modify config.py.

```
(config.py)
pre_trained_model_name = 'bert-base-cased'
num_epochs = 1
batch_size = 10
lr = 1e-5
device = 1
false_num = 3
val_fine_tuned_epo = 1
length_sentence_A = 300
TEST_BATCH_SIZE = 5

```
