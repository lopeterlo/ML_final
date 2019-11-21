import sys
import pandas as pd
import re
import numpy as np
import random
import pickle


import math

def preprocessing(category = 'train'):
    df = pd.read_json(f'./data/{category}.json')
    all_df = pd.DataFrame(columns = ['A', 'B', 'label'])
    special_token = "[\.\!\/_,$%^*()+\"\<>?:]+|[+——！，。？、~@#￥%……&*（）]+"
    for j in range(len(df)):
        input_df = pd.DataFrame(columns = ['A', 'B', 'label'])
        exp = df.iloc[j]
        utterance_mapping = dict()
        ## previous dialogue
        length = len(exp['messages-so-far'])
        data = list(map(lambda x : re.sub(special_token, "",x['utterance']), list(exp['messages-so-far'])))
        A = ''
        for i in range(len(data)):
            A += data[i]

        ## answer part and answers mapping
        utterance_mapping[exp['options-for-correct-answers'][0]['utterance']] = exp['options-for-correct-answers'][0]['candidate-id']
        answer_data = re.sub(special_token, "", exp['options-for-correct-answers'][0]['utterance'])

        if len(A)> 512:
            A = A[-512:]

        input_df['A'] = [A]
        input_df['B'] = [answer_data]
        input_df['label'] = [1]
        print('finished previous dialogue ', end = ',')

        correct_index = list(map(lambda x:  1 if x['candidate-id'] == exp['options-for-correct-answers'][0]['candidate-id'] else 0, exp['options-for-next']))
        correct_index = correct_index.index(1) 


        ## False Label
        false_example_num = 2
        size = len(exp['options-for-next'])
    #     print(size)
        index = [k for k in range(size)]
        random.shuffle(index)
        if correct_index in index:
            index = list(set(index) - set([correct_index]))
        index = index[:false_example_num]
        false_sentence = list()
        for i in index:
            false_sentence.append(exp['options-for-next'][i]['utterance'])
        
        false_sentence = list(map(lambda x : re.sub(special_token, "",x), false_sentence))
        wrong_answer = pd.DataFrame()
        wrong_answer['A'] = [A for i in range(false_example_num)]
        wrong_answer['B'] = false_sentence
        wrong_answer['label'] = [0 for i in range(false_example_num)]
        input_df = input_df.append(wrong_answer).reset_index(drop= True)
        input_df = input_df.replace({'': np.nan}).reset_index(drop = True)
        input_df = input_df.dropna().reset_index(drop= True)
        all_df = all_df.append(input_df).reset_index(drop= True)
        print(f'finished {j} loop / {len(df)}')
    all_df.to_csv(f'small_{category}_df.csv')

def main(argv, arc):
    preprocessing('train')
    preprocessing('valid')

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
