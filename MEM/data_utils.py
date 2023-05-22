import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from IPython.core.debugger import set_trace
import random as rand
import spacy
import pdb


from torch.utils.data import TensorDataset, DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0')
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

tok = spacy.load('en_core_web_sm')


def get_vocabulary(vocab_file):
    return vocab_file
    
def preprocess_explanations_arr(explanations_arr, max_length, existing_vocab=None):
    word2index = None
    
    tokenized_explanations = tokenize_explanations(explanations_arr)
    if existing_vocab is not None:
        word2index = existing_vocab
    else:
        counts = get_counter(tokenized_explanations)
        word2index = vocab2index(counts)
    
    encoded_explanations = encoding(tokenized_explanations, word2index, max_length)

    return encoded_explanations, word2index

def get_counter(tokenized_explanations):
    counts = Counter()
    for tokenized_sentence in tokenized_explanations:
        for token in tokenized_sentence:
            counts.update([token])
    return counts

def vocab2index(counts):
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    return vocab2index
            
def encoding(tokenized_explanations, vocab2index, maxLength):
    encoded_explanations = []
    for tokenized_sentence in tokenized_explanations:
        encoded = np.zeros(maxLength, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized_sentence])
        length = min(maxLength, len(enc1))
        encoded[:length] = enc1[:length]
        encoded_explanations.append(encoded)  
    return encoded_explanations

def tokenize_explanations(explanations_arr):
    tokenized_explanations_arr = []
    for sentence in explanations_arr:
        tokenized_explanations_arr.append([token.text for token in tok.tokenizer(str(sentence))])
    return tokenized_explanations_arr 
        
        
def get_padded_data_C4(y_arr, prevB_arr, currB_arr, gameOver_arr, player_arr, explanation_arr):
    y_tensor = torch.Tensor(y_arr)
    prevB_tensor = torch.Tensor(prevB_arr)
    currB_tensor= torch.Tensor(currB_arr)
    gameOver_tensor = torch.unsqueeze(torch.Tensor(gameOver_arr),1)
    player_tensor = torch.unsqueeze(torch.Tensor(player_arr),1)    
    explanation_tensor = torch.Tensor(np.array(explanation_arr))
  
    # 99 padding 
    explanation_tensor =  F.pad(explanation_tensor, (0, 49-explanation_tensor.size(1)), "constant", 99)
    gameOver_tensor = F.pad(gameOver_tensor, (0, 49-gameOver_tensor.size(1)),"constant", 99)
    player_tensor = F.pad(player_tensor, (0, 49-player_tensor.size(1)),"constant", 99)        
    
    ## 0 padding for CNN to get 7x7 shape for boards
    prevB_tensor = F.pad(prevB_tensor, (0, 49-prevB_tensor.size(1)),"constant", 0)
    currB_tensor = F.pad(currB_tensor, (0, 49-currB_tensor.size(1)),"constant", 0)
    
    return y_tensor, prevB_tensor, currB_tensor, gameOver_tensor, player_tensor, explanation_tensor

def get_padded_data_LL(y_arr, prevS1_arr, currS_arr, game_over_arr, explanation_arr):
    
    y_tensor = torch.Tensor(y_arr)
    prevS1_tensor = torch.Tensor(prevS1_arr) # 1x8
    currS_tensor= torch.Tensor(currS_arr) # 1x8
    currS_tensor = currS_tensor.squeeze(1)
    currS_tensor = currS_tensor.squeeze(1)
    gameOver_tensor = torch.unsqueeze(torch.Tensor(game_over_arr),1)
    explanation_tensor = torch.Tensor(np.array(explanation_arr)) #1x30
 
    #99 padding 
    gameOver_tensor = F.pad(gameOver_tensor, (0, 30-gameOver_tensor.size(1)), "constant", 99)
    prevS1_tensor = F.pad(prevS1_tensor, (0, 30-prevS1_tensor.size(1)), "constant", 99)
    currS_tensor = F.pad(currS_tensor, (0, 30-currS_tensor.size(1)), "constant", 99)
    explanation_tensor =  F.pad(explanation_tensor, (0, 30-explanation_tensor.size(1)), "constant", 99)
        
    return y_tensor, prevS1_tensor, currS_tensor, gameOver_tensor, explanation_tensor
    
def split_dataset(x_tensor, y_tensor, batch_size):

    #creaate 60,20,20 split
    x_train_tensor = x_tensor[0:math.ceil(x_tensor.size(0)*0.6)]
    y_train_tensor = y_tensor[0:math.ceil(x_tensor.size(0)*0.6)]
    valid_end_index = math.ceil(x_tensor.size(0)*0.6)+math.ceil(x_tensor.size(0)*0.2)
    x_valid_tensor = x_tensor[math.ceil(x_tensor.size(0)*0.6):valid_end_index]
    y_valid_tensor = y_tensor[math.ceil(x_tensor.size(0)*0.6):valid_end_index]
    x_test_tensor = x_tensor[valid_end_index:]
    y_test_tensor = y_tensor[valid_end_index:]

    train_dataset = TensorDataset(x_train_tensor,y_train_tensor) 
    valid_dataset = TensorDataset(x_valid_tensor,y_valid_tensor) 
    test_dataset = TensorDataset(x_test_tensor,y_test_tensor) 


    training_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return training_loader, validation_loader, test_loader, x_test_tensor, y_test_tensor


def get_data_arrs_LL(key):
    y_arr = []
    prevS1_arr = []
    currS_arr = []
    game_over_arr = []
    explanation_arr = []
    
    if(key == "LL_data"):
        y_arr = np.load("datasets/LL_data/y_arr.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL_data/prevState1.npy", allow_pickle=True)
        currS_arr = np.load("datasets/LL_data/currentState.npy", allow_pickle=True)
        game_over_arr = np.load("datasets/LL_data/gameOver.npy", allow_pickle=True)
        explanation_arr = np.load("datasets/LL_data/explanation.npy", allow_pickle=True)
       
    return y_arr, prevS1_arr, currS_arr, game_over_arr, explanation_arr

def get_data_arrs_C4(key):
    concept_arr = []
    action_row_col_arr = []
    game_over_arr = []
    player_arr = []
    explanation_arr = []   

    if(key == "C4_data"):
        y_arr = np.load("datasets/C4_data/y.npy", allow_pickle=True)
        prevB_arr = np.load("datasets/C4_data/prevState1.npy", allow_pickle=True)
        currB_arr = np.load("datasets/C4_data/currentState.npy", allow_pickle=True)
        game_over_arr = np.load("datasets/C4_data/gameOver.npy", allow_pickle=True)
        player_arr = np.load("datasets/C4_data/player.npy", allow_pickle=True)
        explanation_arr = np.load("datasets/C4_data/explanation.npy", allow_pickle=True)
       
                                 
    return y_arr, prevB_arr, currB_arr, game_over_arr, player_arr, explanation_arr

# def combine_files():
#     y_arr = np.load("datasets/C4_data/fixed_y_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True)
#     prevB_arr = np.load("datasets/C4_data/fixed_prevBoard_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True)
#     currB_arr = np.load("datasets/C4_data/fixed_currentBoard_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True)
#     game_over_arr = np.load("datasets/C4_data/fixed_gameOver_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True)
#     player_arr = np.load("datasets/C4_data/fixed_player_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True)
#     explanation_arr = np.load("datasets/C4_data/fixed_explanation_states_RL_Pipeline_with_MEM_extraClassses_perturbed.npy", allow_pickle=True) 
    
#     y_arr2 = np.load("datasets/C4_data/fixed_y_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
#     prevB_arr2 = np.load("datasets/C4_data/fixed_prevBoard_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
#     currB_arr2 = np.load("datasets/C4_data/fixed_currentBoard_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
#     game_over_arr2 = np.load("datasets/C4_data/fixed_gameOver_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
#     player_arr2 = np.load("datasets/C4_data/fixed_player_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
#     explanation_arr2 = np.load("datasets/C4_data/fixed_explanation_states_RL_Pipeline_with_GT_extraClassses_perturbed.npy", allow_pickle=True)
    
    
#     new_y = np.concatenate((y_arr,y_arr2),axis=0)
#     new_prevB = np.concatenate((prevB_arr,prevB_arr2),axis=0)
#     new_currB = np.concatenate((currB_arr,currB_arr2),axis=0)
#     new_game_over = np.concatenate((game_over_arr,game_over_arr2),axis=0)
#     new_player = np.concatenate((player_arr,player_arr2),axis=0)
#     new_explanation = np.concatenate((explanation_arr,explanation_arr2),axis=0)
        
#     with open('combined_c4/y.npy', 'wb') as f:
#         np.save(f, new_y)
#     with open('combined_c4/prevState1.npy', 'wb') as f:
#         np.save(f, new_prevB)
#     with open('combined_c4/currentState.npy', 'wb') as f:
#         np.save(f, new_currB)
#     with open('combined_c4/gameOver.npy', 'wb') as f:
#         np.save(f, new_game_over)
#     with open('combined_c4/player.npy', 'wb') as f:
#         np.save(f, new_player)
#     with open("combined_c4/explanation.npy", 'wb') as f:
#         np.save(f, new_explanation)
    
# combine_files()
