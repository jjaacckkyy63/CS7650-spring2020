import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
random.seed(1)

def prepare_sequence(sent, idx_mapping):
    idxs = [idx_mapping[word] for word in sent]
    return torch.tensor(idxs, dtype=torch.long)

def train(epoch, model, loss_function, optimizer, training_data, val_data, word_to_idx, tag_to_idx, logging, pretrained=False):
    train_loss = 0
    train_examples = 0
    for sentence, tags in training_data:
        
        # zero the gradient
        model.zero_grad()
        # Prepare sentence into indexs
        if pretrained:
            sentence_in = sentence
        else:
            sentence_in = prepare_sequence(sentence, word_to_idx)
        # Prepare tag into indexs
        targets = prepare_sequence(tags, tag_to_idx)
        # predictions for the tags of sentence
        tag_scores = model(sentence_in)
        
        loss = loss_function(tag_scores, targets)
        loss.backward()        
        optimizer.step()
        
        train_loss += loss.cpu().detach().numpy()
        train_examples += len(targets.cpu().detach().numpy())
    
    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate(model, loss_function, optimizer, val_data, word_to_idx, tag_to_idx)
    
    with open(logging, 'a') as f:
        f.write("{:.4f}\t{:.4f}\t{:.0f}\n".format(avg_train_loss, avg_val_loss, val_accuracy))
        
    
    print("Epoch: {}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val Accuracy: {:.0f}".format(epoch, 
                                                                      avg_train_loss, 
                                                                      avg_val_loss,
                                                                      val_accuracy))

def evaluate(model, loss_function, optimizer, val_data, word_to_idx, tag_to_idx, pretrained=False):
    val_loss = 0
    correct = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            # Prepare sentence into indexs
            if pretrained:
                sentence_in = sentence
            else:
                sentence_in = prepare_sequence(sentence, word_to_idx)
            # Prepare tag into indexs
            targets = prepare_sequence(tags, tag_to_idx)
            # predictions for the tags of sentence
            tag_scores = model(sentence_in)
            # get the prediction results
            _, preds = torch.max(tag_scores, 1)
            loss = loss_function(tag_scores, targets)
            
            val_loss += loss.cpu().detach().numpy()
            correct += (torch.sum(preds == torch.LongTensor(targets)).cpu().detach().numpy())
            val_examples += len(targets.cpu().detach().numpy())

    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy


