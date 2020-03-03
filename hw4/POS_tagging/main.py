import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torchtext as text
from torch.utils.tensorboard import SummaryWriter

import time

from models import BasicPOSTagger, BiLSTMPOSTagger
from train import prepare_sequence, train, evaluate
from utils import load_tag_data, load_txt_data

random.seed(1)

train_sentences, train_tags = load_tag_data('train.txt')
test_sentences = load_txt_data('test.txt')

unique_tags = set([tag for tag_seq in train_tags for tag in tag_seq])


train_val_data = list(zip(train_sentences, train_tags))
random.shuffle(train_val_data)
split = int(0.8 * len(train_val_data))
training_data = train_val_data[:split]
val_data = train_val_data[split:]

print("Train Data: ", len(training_data))
print("Val Data: ", len(val_data))
print("Test Data: ", len(test_sentences))
print("Total tags: ", len(unique_tags))

word_to_idx = {}
for sent in train_sentences:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

for sent in test_sentences:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
            
tag_to_idx = {}
for tag in unique_tags:
    if tag not in tag_to_idx:
        tag_to_idx[tag] = len(tag_to_idx)

idx_to_tag = {}
for tag in tag_to_idx:
    idx_to_tag[tag_to_idx[tag]] = tag

print("Total tags", len(tag_to_idx))
print("Vocab size", len(word_to_idx))


# init parameters
EMBEDDING_DIM = 200
HIDDEN_DIM = 16
LEARNING_RATE = 0.01
LSTM_LAYERS = 3
DROPOUT = 0.2
EPOCHS = 50
BIDIRECTIONAL = True
pretrained = True

logging = "bilstm_exp_1.txt"
with open(logging, 'w+') as f:
    print("open", logging)

# Initialize the model, optimizer and the loss function
model = BiLSTMPOSTagger(embedding_dim=EMBEDDING_DIM, 
                        hidden_dim=HIDDEN_DIM,
                        vocab_size = len(word_to_idx), 
                        tagset_size = len(tag_to_idx),
                        lstm_layers = LSTM_LAYERS, 
                        dropout_rate = DROPOUT, 
                        bidirectional = BIDIRECTIONAL)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS + 1): 
    start = time.time()
    train(epoch, model, loss_function, optimizer, 
          training_data, val_data, word_to_idx, tag_to_idx, 
          logging, pretrained=pretrained)
    print(f"Time used for Epoch{epoch}: ",time.time() - start)

