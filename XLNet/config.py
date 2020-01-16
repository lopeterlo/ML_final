pre_trained_model_name = 'xlnet-base-cased'
num_epochs = 1
batch_size = 10
lr = 1e-5
device = 1
false_num = 3
val_fine_tuned_epo = 1
length_sentence_A = 300
model_type = f'xlnetSC_f{false_num}_lr{lr}_A{length_sentence_A}'
model_name = f'xlnet_model_{model_type}'


TEST_BATCH_SIZE = 5
