# ML_final

### Machine learning fall 2019 final project -  Dialogue Modeling Problem

![image](https://github.com/lopeterlo/ML_final/blob/master/pic/Task_description.png)
![image](https://github.com/lopeterlo/ML_final/blob/master/pic/data_format.png)
![image](https://github.com/lopeterlo/ML_final/blob/master/pic/output_format.png)


### data preprocessing:

python3 data_preprocessing.py


### Fine-tuning with Bert pretrained model:

bash train.sh train_data_path.csv valid_data_path.csv valid_train_path.csv 


### Testing:

bash test_data_path.csv output_path.csv
