import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import utils_LL
import pdb
from IPython.core.debugger import set_trace
import random as rand
import json

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

def get_test_MLP(batch_size=2):
    y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr = utils_LL.get_data_arrs("LL_test_WellTrained6000")
    
    _, prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, class_tensor, action_tensor, gameOver_tensor = utils_LL.get_padded_data("FCL", y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr)
    
    #explicit assignment to make clear that in the MLP setting, y_tensor is the class nums arr associated with each state
    y_tensor = class_tensor.clone().detach()                                                                                     
    #create X dataset with 60-20-20 split
    x_tensor = torch.stack((prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, action_tensor, gameOver_tensor),axis=1)
    return x_tensor, y_tensor

def get_data_MEM_models(batch_size=2):
    ### Getting MEM used RL PIPELINE DATA
#     with open('vocab_LL.json', 'r') as openfile:
#         vocab = json.load(openfile)
    y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr, _ = utils_LL.get_data_arrs("DC1_GO_with_power_LS")
    
#     processed_explanations_arr, word2index = utils_LL.preprocess_explanations_arr(explanation_arr, vocab)
    
    y_tensor, prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, class_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor = utils_LL.get_padded_data("FCL", y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr)
    
#     vocab = utils_LL.get_vocabulary(word2index)
#     with open("vocab_LL.json", "w") as outfile:
#         json.dump(vocab, outfile)
    
#     y_arr_EXTRA, prevS1_arr_EXTRA, prevS2_arr_EXTRA, prevS3_arr_EXTRA, currS_arr_EXTRA, class_arr_EXTRA, action_arr_EXTRA, game_over_arr_EXTRA, concept_encoding_arr_EXTRA = utils_LL.get_data_arrs("DC1_GO_with_power")   
    
#     y_tensor_EXTRA, prevS1_tensor_EXTRA, prevS2_tensor_EXTRA, prevS3_tensor_EXTRA, currS_tensor_EXTRA, class_tensor_EXTRA, action_tensor_EXTRA, gameOver_tensor_EXTRA, concept_encoding_tensor_EXTRA = utils_LL.get_padded_data("FCL", y_arr_EXTRA, prevS1_arr_EXTRA, prevS2_arr_EXTRA, prevS3_arr_EXTRA, currS_arr_EXTRA, class_arr_EXTRA, action_arr_EXTRA, game_over_arr_EXTRA, concept_encoding_arr_EXTRA)
                                      
#     # create datasets (combine GT RL and EXTRA pipeline data)
#     y_tensor = torch.cat((y_tensor, y_tensor_EXTRA))
    
    x_tensor = torch.stack((prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor),axis=1)
#     x_tensor_EXTRA = torch.stack((prevS1_tensor_EXTRA, prevS2_tensor_EXTRA, prevS3_tensor_EXTRA, currS_tensor_EXTRA, action_tensor_EXTRA, gameOver_tensor_EXTRA, concept_encoding_tensor_EXTRA),axis=1)
    
#     x_tensor = torch.cat((x_tensor, x_tensor_EXTRA))

    x_train_tensor = x_tensor[0:math.ceil(x_tensor.size(0)*0.6)]
#     x_train_tensor = torch.cat((x_train_tensor, x_tensor_EXTRA))
    y_train_tensor = y_tensor[0:math.ceil(x_tensor.size(0)*0.6)]
#     y_train_tensor = torch.cat((y_train_tensor, y_tensor_EXTRA))

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
    
    # single dataset 
#     #create X dataset with 60-20-20 split
#     x_tensor = torch.stack((prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor),axis=1)
    
    type_= ""
    if(type_=="NLP"):
        return training_loader, validation_loader, test_loader, x_test_tensor, y_test_tensor, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, vocab
    else:
        return training_loader, validation_loader, test_loader, x_test_tensor, y_test_tensor, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor
    

def get_test_MEM(batch_size=2):
    with open('vocab_LL.json', 'r') as openfile:
        vocab = json.load(openfile)
    y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr, explanation_arr = utils_LL.get_data_arrs("DC1_GO_with_power_LS")
    
#     processed_explanations_arr, _ = utils_LL.preprocess_explanations_arr(explanation_arr, vocab)
    
    y_tensor, prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, class_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor = utils_LL.get_padded_data("FCL", y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr)
                                                                                 
    #create X dataset with 60-20-20 split
    x_tensor = torch.stack((prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor),axis=1)
    
    return x_tensor, y_tensor
      
    
def get_data_MLP_models(batch_size=2):    
    
    ### Getting MEM used RL PIPELINE DATA
    y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr = utils_LL.get_data_arrs("LL_DC1_RL_Pipe")
    
    _, prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, class_tensor, action_tensor, gameOver_tensor = utils_LL.get_padded_data("FCL", y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr)
    
    #explicit assignment to make clear that in the MLP setting, y_tensor is the class nums arr associated with each state
    y_tensor = class_tensor.clone().detach()                                                                                     
    #create X dataset with 60-20-20 split
    x_tensor = torch.stack((prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, action_tensor, gameOver_tensor),axis=1)
    
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

                   
    return training_loader, validation_loader, test_loader, x_test_tensor, y_test_tensor, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor


