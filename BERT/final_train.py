import sys
import pickle
import pandas as pd

from final_common import DialogueDataset, bert_model

from config import  pre_trained_model_name, num_epochs, batch_size, lr, device, false_num, val_fine_tuned_epo, length_sentence_A, model_type, model_name


def main(argv, arc):
    train_path = argv[1]
    val_path = argv[2]
    val_train_path = argv[3]

    train_df = pd.read_csv(train_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(['Unnamed: 0'], axis =1)
    
    val_df = pd.read_csv(val_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in val_df.columns:
        val_df = val_df.drop(['Unnamed: 0'], axis =1)

    val_train_df = pd.read_csv(val_train_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in val_train_df.columns:
        val_train_df = val_train_df.drop(['Unnamed: 0'], axis =1)

    model = bert_model(epoch = num_epochs, batch_size = batch_size, lr = lr)
    model.fit_and_train(train_df, val_df, val_train_df, require_grad = True)
    # with open(f'./model/{model_name}_last_epo', 'wb') as output:
    #     pickle.dump(model, output)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))