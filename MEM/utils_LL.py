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


def get_vocabulary(vocab1):
    return vocab1

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

def tokenize_explanations(explanations_arr):
    tok = spacy.load('en_core_web_sm')
    tokenized_explanations_arr = []
    for sentence in explanations_arr:
        tokenized_explanations_arr.append([token.text for token in tok.tokenizer(str(sentence))])
    return tokenized_explanations_arr 

def encoding(tokenized_explanations, vocab2index, maxLength=30):
    encoded_explanations = []
    len_tokenized = 0
    for tokenized_sentence in tokenized_explanations:
        encoded = np.zeros(maxLength, dtype=int)
        if(len_tokenized < len(tokenized_sentence)):
            len_tokenized = len(tokenized_sentence)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized_sentence])
        length = min(maxLength, len(enc1))
        encoded[:length] = enc1[:length]
        encoded_explanations.append(encoded) 
    return encoded_explanations

def preprocess_explanations_arr(explanation_arr, existing_vocab=None):    
    word2index = None
    
    tokenized_explanations = tokenize_explanations(explanation_arr)
    if existing_vocab is not None:
        word2index = existing_vocab
    else:
        counts = get_counter(tokenized_explanations)
        word2index = vocab2index(counts)
    
    encoded_explanations = encoding(tokenized_explanations, word2index)
    
    return encoded_explanations, word2index
    

def get_padded_data(key, y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr):
    concept_encoding_tensor = torch.Tensor(concept_encoding_arr) #1x13
    y_tensor = torch.Tensor(y_arr)
    prevS1_tensor = torch.Tensor(prevS1_arr) # 1x8
    prevS2_tensor = torch.Tensor(prevS2_arr) # 1x8
    prevS3_tensor = torch.Tensor(prevS3_arr) # 1x8
    currS_tensor= torch.Tensor(currS_arr) # 1x8
    
    currS_tensor = currS_tensor.squeeze(1)
    currS_tensor = currS_tensor.squeeze(1)
    action_tensor = torch.unsqueeze(torch.Tensor(action_arr),1) 
    gameOver_tensor = torch.unsqueeze(torch.Tensor(game_over_arr),1)
    class_tensor = torch.Tensor(class_arr)
#     explanation_tensor = torch.Tensor(np.array(explanation_arr)) #1x30

        
    if(key == "FCL"): #FIX THE PADDING
        #99 padding for FCL
        action_tensor = F.pad(action_tensor, (0, 30-action_tensor.size(1)), "constant", 99)
        gameOver_tensor = F.pad(gameOver_tensor, (0, 30-gameOver_tensor.size(1)), "constant", 99)
        prevS1_tensor = F.pad(prevS1_tensor, (0, 30-prevS1_tensor.size(1)), "constant", 99)
        prevS2_tensor = F.pad(prevS2_tensor, (0, 30-prevS2_tensor.size(1)), "constant", 99)
        prevS3_tensor = F.pad(prevS3_tensor, (0, 30-prevS3_tensor.size(1)), "constant", 99)
        currS_tensor = F.pad(currS_tensor, (0, 30-currS_tensor.size(1)), "constant", 99)
#         explanation_tensor =  F.pad(explanation_tensor, (0, 30-explanation_tensor.size(1)), "constant", 99)
        concept_encoding_tensor = F.pad(concept_encoding_tensor, (0, 30-concept_encoding_tensor.size(1)), "constant", 99)
        
    elif(key == "LSTM"):
        # 99 padding for LSTM
        action_tensor = F.pad(action_tensor, (0, 20-action_tensor.size(1)), "constant", 99)
        gameOver_tensor = F.pad(gameOver_tensor, (0, 20-gameOver_tensor.size(1)), "constant", 99)
    
    return y_tensor, prevS1_tensor, prevS2_tensor, prevS3_tensor, currS_tensor, class_tensor, action_tensor, gameOver_tensor, concept_encoding_tensor

        
def get_data_arrs(key):
    y_arr = []
    prevS1_arr = []
    prevS2_arr = []
    prevS3_arr = []
    currS_arr = []
    class_arr = []
    action_arr = []
    game_over_arr = []
    concept_encoding_arr = []   
    explanation_arr = []
    
    if(key == "DC1_with_power_LS_explanation"):
        y_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/y_arr_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/prevState1_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/DC1_with_power_LS_explanation/prevState2_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/DC1_with_power_LS_explanation/prevState3_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/currentState_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/conceptClass_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/action_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/gameOver_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/conceptEncoding_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
        explanation_arr = np.load("datasets/LL/DC1_with_power_LS_explanation/explanation_LL_DC1_RL_Pipe_with_power_LS_explanation.npy", allow_pickle=True)
       
    #this is the good one
    if(key == "DC1_GO_with_power_LS"):
        y_arr = np.load("datasets/LL/DC1_GO_with_power_LS/y_arr_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/DC1_GO_with_power_LS/prevState1_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/DC1_GO_with_power_LS/prevState2_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/DC1_GO_with_power_LS/prevState3_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/DC1_GO_with_power_LS/currentState_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/DC1_GO_with_power_LS/conceptClass_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/DC1_GO_with_power_LS/action_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/DC1_GO_with_power_LS/gameOver_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/DC1_GO_with_power_LS/conceptEncoding_LL_DC1_RL_Pipe_GO_power_LS.npy", allow_pickle=True)
                                       
    if(key == "GO_scenarios1"):
        y_arr = np.load("datasets/LL/GO_scenarios1/y_arr_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/GO_scenarios1/prevState1_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/GO_scenarios1/prevState2_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/GO_scenarios1/prevState3_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/GO_scenarios1/currentState_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/GO_scenarios1/conceptClass_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/GO_scenarios1/action_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/GO_scenarios1/gameOver_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/GO_scenarios1/conceptEncoding_LL_DC1_RL_Pipe_GO_Scenarios1.npy", allow_pickle=True)

    if(key == "RL_sanity_test_mem"):
        y_arr = np.load("datasets/LL/sanity-MEM/y_arr_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/sanity-MEM/prevState1_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/sanity-MEM/prevState2_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/sanity-MEM/prevState3_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/sanity-MEM/currentState_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/sanity-MEM/conceptClass_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/sanity-MEM/action_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/sanity-MEM/gameOver_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/sanity-MEM/conceptEncoding_LL_DC1_RL_Pipe_sanity.npy", allow_pickle=True)
   
        
    if(key == "DC1_RL_Pipe_Perturbed"):
        y_arr = np.load("datasets/LL/RL_pipeline_perturbed/y_arr_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/RL_pipeline_perturbed/prevState1_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/RL_pipeline_perturbed/prevState2_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/RL_pipeline_perturbed/prevState3_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/RL_pipeline_perturbed/currentState_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/RL_pipeline_perturbed/conceptClass_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/RL_pipeline_perturbed/action_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/RL_pipeline_perturbed/gameOver_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/RL_pipeline_perturbed/conceptEncoding_LL_DC1_RL_Pipe_Perturbed.npy", allow_pickle=True)
    if(key == "DC2_RL_Pipe_Test"):
        y_arr = np.load("datasets/LL/testRLPipeline-MEM/y_arr_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/testRLPipeline-MEM/prevState1_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/testRLPipeline-MEM/prevState2_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/testRLPipeline-MEM/prevState3_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)  
        currS_arr = np.load("datasets/LL/testRLPipeline-MEM/currentState_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/testRLPipeline-MEM/conceptClass_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/testRLPipeline-MEM/action_LL_DC2_RL_Pipe_test.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/testRLPipeline-MEM/gameOver_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
        concept_encoding_arr = np.load("datasets/LL/testRLPipeline-MEM/conceptEncoding_LL_DC2_RL_Pipe_test.npy", allow_pickle=True)
    if(key == "LL_DC1_RL_Pipe"):
        y_arr = np.load("datasets/LL/RL_pipeline_unperturbed/y_arr_LL_DC1_inRLPipe.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/RL_pipeline_unperturbed/prevState1_LL_DC1_inRLPipe.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/RL_pipeline_unperturbed/prevState2_LL_DC1_inRLPipe.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/RL_pipeline_unperturbed/prevState3_LL_DC1_inRLPipe.npy", allow_pickle=True)        
        currS_arr = np.load("datasets/LL/RL_pipeline_unperturbed/currentState_LL_DC1_inRLPipe.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/RL_pipeline_unperturbed/conceptClass_LL_DC1_inRLPipe.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/RL_pipeline_unperturbed/action_LL_DC1_inRLPipe.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/RL_pipeline_unperturbed/gameOver_LL_DC1_inRLPipe.npy", allow_pickle=True)
        
    if(key == "LL_test_RL_Pipe"):
        y_arr = np.load("datasets/LL/test_RLPipeline/y_arr_LL_DC2_inRLPipe.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/test_RLPipeline/prevState1_LL_DC2_inRLPipe.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/test_RLPipeline/prevState2_LL_DC2_inRLPipe.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/test_RLPipeline/prevState3_LL_DC2_inRLPipe.npy", allow_pickle=True)        
        currS_arr = np.load("datasets/LL/test_RLPipeline/currentState_LL_DC2_inRLPipe.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/test_RLPipeline/conceptClass_LL_DC2_inRLPipe.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/test_RLPipeline/action_LL_DC2_inRLPipe.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/test_RLPipeline/gameOver_LL_DC2_inRLPipe.npy", allow_pickle=True)
        
    if(key == "LL_test_WellTrained6000"):
        y_arr = np.load("datasets/LL/test_wellTrained_6000/y_arr_LL_WellTrained_6000.npy", allow_pickle=True)
        prevS1_arr = np.load("datasets/LL/test_wellTrained_6000/prevState1_LL_WellTrained_6000.npy", allow_pickle=True)
        prevS2_arr= np.load("datasets/LL/test_wellTrained_6000/prevState2_LL_WellTrained_6000.npy", allow_pickle=True)
        prevS3_arr= np.load("datasets/LL/test_wellTrained_6000/prevState3_LL_WellTrained_6000.npy", allow_pickle=True)        
        currS_arr = np.load("datasets/LL/test_wellTrained_6000/currentState_LL_WellTrained_6000.npy", allow_pickle=True)
        class_arr = np.load("datasets/LL/test_wellTrained_6000/conceptClass_LL_WellTrained_6000.npy", allow_pickle=True)
        action_arr = np.load("datasets/LL/test_wellTrained_6000/action_LL_WellTrained_6000.npy", allow_pickle=True) 
        game_over_arr = np.load("datasets/LL/test_wellTrained_6000/gameOver_LL_WellTrained_6000.npy", allow_pickle=True)
    
    return y_arr, prevS1_arr, prevS2_arr, prevS3_arr, currS_arr, class_arr, action_arr, game_over_arr, concept_encoding_arr, explanation_arr
