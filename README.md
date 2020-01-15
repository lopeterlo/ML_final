# ML_final

## Machine learning fall 2019 final project -  Dialogue Modeling Problem

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
### Download dataset
```
./download_dataset.sh
```
Three files (**train.json, valid.json, test.json**) would be downloaded in **./data** folder.

### data preprocessing:

```
python3 data_preprocessing.py
```
This will generate structured data in **./struct_data** folder.

### Fine-tuning with Bert pretrained model:

```
bash train.sh [train_data] [valid_data] [valid_train]
```

### Testing:
```
bash [test_data] [output]
```
