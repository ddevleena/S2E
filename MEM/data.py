import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from IPython.core.debugger import set_trace
import random as rand
import json

import data_utils


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


def get_data_LL(batch_size=1):
    
    max_length = 30
    y_arr, prevS1_arr, currS_arr, game_over_arr, explanation_arr = data_utils.get_data_arrs_LL("LL_data")  
    processed_explanations_arr, word2index = data_utils.preprocess_explanations_arr(explanation_arr, max_length)
    y_tensor, prevS1_tensor, currS_tensor, gameOver_tensor, explanation_tensor = data_utils.get_padded_data_LL(y_arr, prevS1_arr, currS_arr, game_over_arr, processed_explanations_arr)
    
    #use this code if retraining and wanting to generate new vocab file
#     vocab = data_utils.get_vocabulary(word2index)
#     with open("vocab_LL.json", "w") as outfile:
#         json.dump(vocab, outfile)
    
    x_tensor = torch.stack((prevS1_tensor, currS_tensor, gameOver_tensor, explanation_tensor),axis=1)
        
    #Load the existing vocab
    with open('vocab_LL.json', 'r') as openfile:
        vocab = json.load(openfile)

    training_loader, validation_loader, test_loader, x_test, y_test = data_utils.split_dataset(x_tensor, y_tensor, batch_size)

    return training_loader, validation_loader, test_loader, x_test, y_test, vocab
    
    
def get_data_C4(batch_size=1):
    max_length = 11

    y_arr, prevB_arr, currB_arr, game_over_arr, player_arr, explanation_arr = data_utils.get_data_arrs_C4("C4_data")
    processed_explanations_arr, word2index = data_utils.preprocess_explanations_arr(explanation_arr, max_length)
    
    y_tensor, prevB_tensor, currB_tensor, gameOver_tensor, player_tensor, explanation_tensor = data_utils.get_padded_data_C4(y_arr, prevB_arr, currB_arr, game_over_arr, player_arr, processed_explanations_arr)
    
    #use this code if retraining and wanting to generate new vocab file
#     vocab = data_utils.get_vocabulary(word2index)
#     with open("vocab_C4.json", "w") as outfile:
#         json.dump(vocab, outfile)
   
            
    x_tensor = torch.stack((prevB_tensor, currB_tensor, gameOver_tensor, player_tensor, explanation_tensor),axis=1)
            
    training_loader, validation_loader, test_loader, x_test, y_test = data_utils.split_dataset(x_tensor, y_tensor, batch_size)

#     #Load the existing vocab
    with open('vocab_C4.json', 'r') as openfile:
        vocab = json.load(openfile)

    return training_loader, validation_loader, test_loader, x_test, y_test, vocab
    

