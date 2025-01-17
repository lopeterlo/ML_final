import sys
import pickle
import pandas as pd

import torch
from transformers import BertForSequenceClassification
from transformers import BertModel, BertTokenizer, AdamW

from final_common import DialogueDataset, bert_model
from config import  pre_trained_model_name
from config import device

def main(argv, arc):
    assert len(argv) == 4, 'input should be :test_data, output_path, model_path '
    test_path = argv[1]
    model_name = argv[2]
    output_path = argv[3]
    test_df = pd.read_csv(test_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(['Unnamed: 0'], axis =1)

    print(len(test_df),end = '\n')
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    testset = DialogueDataset(test_df, 'test', tokenizer=tokenizer)


    # first way
    # with open(f'./model/{model_name}', 'rb') as input_model:
    #     model = pickle.load(input_model)

    # second way
    NUM_LABELS = 2
    
    model = bert_model()
    model.model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=NUM_LABELS)
    # model.model = BertForNextSentencePrediction.from_pretrained(pre_trained_model_name)
    model.model.load_state_dict(torch.load(f'{model_name}', map_location= f'cuda:{device}'))
    print(model.val_accu_list)

    preds = model.predict(testset)
    test_df['prob'] = preds
    groups = test_df.groupby('question')
    ans = []
    for index, data in groups:
        if 'candidate_id' in test_df.columns:
            ans.append(data.loc[data['prob'].idxmax(),'candidate_id'])
        else:
            ans.append(data.loc[data['prob'].idxmax(),'B'])

    pred_df = pd.DataFrame()
    pred_df['id'] = [f'{80000 + i}' for i in range(0, len(ans))]
    pred_df['candidate-id'] = ans
    pred_df.to_csv(output_path, index = False)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))