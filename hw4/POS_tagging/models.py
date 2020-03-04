import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torchtext as text
random.seed(1)

class BasicPOSTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BasicPOSTagger, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        tag_scores = None

        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

class CharPOSTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim, 
                 char_hidden_dim, char_size, vocab_size, tagset_size, MAX_WORD_LEN):
        super(CharPOSTagger, self).__init__()

        # word embedding
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_word = nn.LSTM(embedding_dim, self.hidden_dim)
        
        # char embedding
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim)
        self.lstm_char = nn.LSTM(char_embedding_dim, self.char_hidden_dim)
        
        # combine the word / character
        self.overall_hidden_dim = hidden_dim + MAX_WORD_LEN * char_hidden_dim
        
        self.hidden2tag = nn.Linear(self.overall_hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        self.char_hidden = self.init_hidden(isChar=True)

    def init_hidden(self, isChar=False):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if isChar:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.char_hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, 1, self.char_hidden_dim)))
        else:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))        

    def forward(self, sentence, chars):
        tag_scores = None

        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm_word(embeds.view(len(sentence), 1, -1), self.hidden)
        
        embeds_char = self.char_embedding(chars)
        char_lstm_out, self.char_hidden = self.lstm_char(embeds_char.view(len(chars), 1, -1), self.char_hidden)
        
        # Remember!!!!!!! You Should re-organized the characters into sentence!!!!!!!!!!
        merge_out = torch.cat((lstm_out.view(len(sentence), -1), char_lstm_out.view(len(sentence), -1)), 1)
        
        tag_space = self.hidden2tag(merge_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        tag_space = self.hidden2tag(merge_out)
        tag_scores = F.log_softmax(tag_scores, dim=1)

        return tag_scores

class BiLSTMPOSTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, lstm_layers, dropout_rate, bidirectional):
        super(BiLSTMPOSTagger, self).__init__()

        self.vec = text.vocab.GloVe(name='6B', dim=embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim, 
                        num_layers=lstm_layers, 
                        dropout = dropout_rate if lstm_layers > 1 else 0,bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, tagset_size)

    def forward(self, sentence):
        tag_scores = None
        embeds = self.vec.get_vecs_by_tokens(sentence, lower_case_backup=True)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
