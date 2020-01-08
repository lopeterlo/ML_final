import sys
import numpy as np
import pickle
import pandas as pd
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data import Dataset
 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchvision
from torchvision import datasets, models, transforms


from transformers import BertForSequenceClassification, BertForNextSentencePrediction, RobertaForSequenceClassification
from transformers import BertModel, BertTokenizer, AdamW, RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup


# pre_trained_model_name = 'bert-base-uncased'
pre_trained_model_name = 'roberta-base'
num_epochs = 3
batch_size = 8
val_batch_size = 8
lr = 1e-6
device = 1
false_num = 3
val_fine_tuned_epo = 1




length_sentence_A = 250
model_type = f'SC_adamw_f{false_num}_valepo{val_fine_tuned_epo}_A{length_sentence_A}'

model_name = f'Roberta_model_{lr}_{num_epochs}_lower_1227_{model_type}'

def log(data):
    with open('log.txt', 'w') as f:
        f.write(str(data))

class DialogueDataset(Dataset):
    def __init__(self, df, mode , tokenizer):
        assert mode in ["train", "test"]  # 
        self.mode = mode
        self.df = df
        self.len = len(self.df)
        self.tokenizer = tokenizer  # transformer 中的 BERT tokenizer
    
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        # 將第一句 tokenize 後加入，並在最後補上分隔符號 [SEP]
        word_pieces = [self.tokenizer.cls_token]
        text_a += self.tokenizer.sep_token
        len_a = len(self.tokenizer.tokenize(text_a, add_prefix_space=True)) 

        all_text = text_a + text_b
        tokens = self.tokenizer.tokenize(all_text, add_prefix_space=True)
        tokens = tokens[:510]

        word_pieces += tokens + [self.tokenizer.sep_token]
        

        # 將整個 token 序列轉換成索引序列，將 tokens 轉成 token id 
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        len_b = len(tokens_tensor.tolist()) - 2 - len_a 
        segments_tensor =[0] * (len_a +1) + [1] * (len_b+1)
        segments_tensor = segments_tensor[:512]
        segments_tensor = torch.tensor(segments_tensor, dtype=torch.long)



        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

class CustomRobertatModel(nn.Module):
    def __init__(self, num_labels = 2, model_name = 'roberta-base'):
        super(CustomRobertatModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.roberta.num_labels = 2
        self.roberta.type_vocab_size = 2
        self.dropout = nn.Dropout(.05)
        self.classifier = nn.Linear(768, num_labels)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)   

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            return outputs
        return logits


class Roberta_model():
    def __init__(self, epoch = 100, batch_size = 32, lr = 1e-4, valset = None):
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss_list = []
        self.val_loss_list = []
        self.val_accu_list = []
        self.lr = lr
        self.model = None
        self.gpu = torch.cuda.is_available()

    def create_mini_batch(self, samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]

        # 測試集有 labels
        if samples[0][2] is not None:
            label_ids = torch.stack([s[2] for s in samples])
        else:
            label_ids = None

        # zero pad 到同一序列長度
        tokens_tensors = pad_sequence(tokens_tensors, 
                                      batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, 
                                        batch_first=True)

        # attention masks，將 tokens_tensors 裡頭不為 zero padding
        # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)

        return tokens_tensors, segments_tensors, masks_tensors, label_ids
        
    
    def fit_and_train(self, train_df, val_df, val_train_df, require_grad):

        NUM_LABELS = 2
        max_value = 0
        best_model = None

        tokenizer = RobertaTokenizer.from_pretrained(pre_trained_model_name, do_lower_case=True)
       
        trainset = DialogueDataset(train_df, "train", tokenizer=tokenizer)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, collate_fn=self.create_mini_batch)

        val_trainset = DialogueDataset(val_train_df, "train", tokenizer=tokenizer)
        val_trainloader = DataLoader(val_trainset, batch_size=self.batch_size, collate_fn=self.create_mini_batch)

        
        valset = DialogueDataset(val_df, 'test', tokenizer = tokenizer)
        valloader = DataLoader(valset, batch_size=val_batch_size, collate_fn=self.create_mini_batch)
        
        config = RobertaConfig.from_pretrained(pre_trained_model_name)
        config.num_labels = 2
        config.type_vocab_size = 2

        model = RobertaForSequenceClassification(config)
        # model = CustomRobertatModel()
        # model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=NUM_LABELS)
        # model = BertForNextSentencePrediction.from_pretrained(pre_trained_model_name)
        # if require_grad:
        #   for param in model.parameters():
        #     param.requires_grad = True
        model.train()

        if self.gpu:
            model = model.cuda(device)
        for epo in range(self.epoch):
            total = 0
            total_loss = 0
            
            # optimizer = AdamW(model.parameters(),
            #       lr = self.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            #       eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            #     )

            optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01 , eps=1e-6)

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(trainloader) * self.epoch

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps = 1000, t_total = total_steps)

            for data in trainloader:
                if self.gpu:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x.type(torch.LongTensor).cuda(device) for x in data]
                else:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x for x in data]
                outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
   # (tensor(0.6968, grad_fn=<NllLossBackward>), tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))
                loss = outputs [0]
# (tensor(0.0086, device='cuda:1', grad_fn=<NllLossBackward>), tensor([[ 2.3423, -2.4149]], device='cuda:1', grad_fn=<AddmmBackward>))

                loss.backward() # calculate gradientopt = torch.optim.SGD(model.parameters(), lr=self.lr,  momentum=0.9)
                # opt = torch.optim.Adam(model.parameters(), lr = self.lr)
                # opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
                # opt.step() #update parameter
                # opt.zero_grad()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                # Clear out the gradients (by default they accumulate)
                model.zero_grad()


                total += len(tokens_tensors)
                total_loss += loss.item() * len(tokens_tensors)

                # outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
                # loss_f = nn.CrossEntropyLoss()
                # loss = loss_f(outputs[0], labels)
                # loss.backward() # calculate gradientopt = torch.optim.SGD(model.parameters(), lr=self.lr,  momentum=0.9)
                # opt = torch.optim.Adam(model.parameters(), lr = self.lr)
                # opt.step() #update parameter
                # opt.zero_grad()
                # total += len(tokens_tensors)
                # total_loss += loss.item() * len(tokens_tensors)

                del data, tokens_tensors, segments_tensors, \
                    masks_tensors, labels

                print(f'Epoch : {epo+1}/{self.epoch} , Training Loss : {loss}', end = '\r')
            self.loss_list.append(total_loss / total)
            print(f'Epoch : {epo+1}/{self.epoch} , Training Loss : {self.loss_list[epo]}', end = ',')

            with open (f'./train_loss_{model_type}.txt', 'w') as f:
                for i in self.loss_list:
                    f.write(str(i)+ '\n')

            model.eval()
            numebr = 0

            ans = []
            with torch.no_grad():
                for data in valloader:
                    if self.gpu:
                        tokens_tensors, segments_tensors, masks_tensors, _ = [x.type(torch.LongTensor).cuda(device) if x is not None else None for x in data]
                    else:
                        tokens_tensors, segments_tensors, masks_tensors, _ = [x for x in data]
                    outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,)
        #      (tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))   
                    values = outputs[0].data[:,1].tolist()
                    ans += values
                    print(f'count : {numebr}', end = '\r')
                    numebr += val_batch_size

                count = 0
                val_len = 0
                val_df['prob'] = ans
                groups = val_df.groupby('question')
                for index, data in groups:
                    val_len += 1
                    if 'candidate_id' in val_df.columns:
                        pred_id = data.loc[data['prob'].idxmax(),'candidate_id']
                        if data.loc[data['prob'].idxmax(),'ans'] == pred_id:
                            count += 1


                val_accu = count / val_len
                if val_accu >= max_value:
                    max_value = val_accu
                    self.model = model
                    best_model = model
                    torch.save(model.state_dict(), f'./model/{model_name}_torch_dict') 
                self.val_accu_list.append(val_accu)

                print(f'Epoch : {epo+1}/{self.epoch}, Validation Accuracy : {self.val_accu_list[epo]}',end = ',')
                with open (f'./val_accu_{model_type}.txt', 'w') as f:
                    for i in self.val_accu_list:
                        f.write(str(i)+ '\n')

        ## Eventually fine tuned with validation data

        for epo in range(val_fine_tuned_epo):
            total = 0
            total_loss = 0

            optimizer = AdamW(best_model.parameters(),
                  lr = self.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
            total_steps = len(val_trainloader) * 1
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps = 10, t_total = total_steps)

            for data in val_trainloader:
                if self.gpu:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x.type(torch.LongTensor).cuda(device) for x in data]
                else:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x for x in data]
                outputs = best_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
                loss = outputs [0]
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()


                total += len(tokens_tensors)
                total_loss += loss.item() * len(tokens_tensors)
                del data, tokens_tensors, segments_tensors, \
                    masks_tensors, labels

        # check if fine tune with validation work
        torch.save(best_model.state_dict(), f'./model/{model_name}_torch_dict_tuned_val') 

    def accu(self, pred, y):
        ret = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                ret += 1
        return ret

    def softmax(self, vec):
        out = Func.softmax(vec, dim=1) # along rows
        values, indexs = out.max(-1)
        return indexs.view(len(indexs),1)
    
    def accu(self, pred, y):
        ret = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                ret += 1
        return ret
    
    def forward(self, x):
        out = Func.softmax(x, dim=1) # along rows
        return out[0][1].tolist()

    
    def predict(self, test_data):
        test_batch_size = 1
        ans = []
        if self.gpu:
            self.model = self.model.cuda(device)
        self.model.eval()
        testloader = DataLoader(test_data, batch_size=test_batch_size, collate_fn=self.create_mini_batch)
        count = 0 
        for x in testloader:
            if self.gpu:
                tokens_tensors, segments_tensors, masks_tensors, _ = [i.cuda(device) if i is not None else i for i in x ]
            else:
                tokens_tensors, segments_tensors, masks_tensors, _= [i for i in x]
            outputs = self.model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors,)
#      (tensor(0.6968, grad_fn=<NllLossBackward>), tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))
            # first method -> batch >1
            #      (tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))   
            values = outputs[0].data[:,1].tolist()
            ans += values
            # second method
            # ans.append(self.forward(outputs[0]))
            # ans.append(outputs[0].tolist()[0][1])
            count+= test_batch_size
            print(f'count : {count}', end = '\r')
        return ans
       
       

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

    model = Roberta_model(epoch = num_epochs, batch_size = batch_size, lr = lr)
    model.fit_and_train(train_df, val_df, val_train_df, require_grad = True)
    with open(f'./model/{model_name}_last_epo', 'wb') as output:
        pickle.dump(model, output)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))