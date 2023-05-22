import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import random as rand
import spacy
import pdb

### This file has a few overlapping utility functions that are present in the MEM data_utils.py and train_test_utils.py in MEM folder 
### Adding this file here to denote some functions used within both connect4.py and lunarlander.py for running the MEM models during RL training

tok = spacy.load('en_core_web_sm')

explanation_list_C4 = ["a generic move not tied to a strategy", "blocks an opponent win and provides center dominance","results in a three in a row and provides center dominance", "blocks an opponent win and results in a three in a row", "results in a three in a row but is blocked from a win", "leads to a three in a row", "provides center dominance", "blocks an opponent win", "leads to a win"]

explanation_list_LL = ["moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves side fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves main fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves main fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground", "encourages lander to land"]

def get_explanation_list(domain):
    if(domain == "C4"):
        return explanation_list_C4
    else:
        return explanation_list_LL

def tokenize_explanations(explanations_arr):
    tokenized_explanations_arr = []
    for sentence in explanations_arr:
        tokenized_explanations_arr.append([token.text for token in tok.tokenizer(str(sentence))])
    return tokenized_explanations_arr 


def encoding(tokenized_explanations, vocab2index, maxLength):
    encoded_explanations = []
    for tokenized_sentence in tokenized_explanations:
        encoded = np.zeros(maxLength, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized_sentence])
        length = min(maxLength, len(enc1))
        encoded[:length] = enc1[:length]
        encoded_explanations.append(encoded)  
    return encoded_explanations


def get_masks_LL(x_prevS1, x_currS, x_gameOver, x_explanation):
    mask_prevS1 = x_prevS1.ne(99)
    mask_currS = x_currS.ne(99)
    mask_gameOver = x_gameOver.ne(99)
    mask_explanation = x_explanation.ne(99)
    
    return mask_prevS1, mask_currS, mask_gameOver, mask_explanation

def get_masks_C4(x_prevBoard,x_currBoard,x_gameOver, x_player, x_explanation):
    mask_prevB = x_prevBoard.ne(99)
    mask_currB = x_currBoard.ne(99)
    mask_gameOver = x_gameOver.ne(99)
    mask_player = x_player.ne(99)
    mask_explanation = x_explanation.ne(99)
    
    return mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation
        


def get_shorthand_concept_LL(predicted_explanationVec):
        #need to go from explanation string to the '-' separated concepts via the dictionary
        class_num = dict_concepts_explanations[predicted_explanationVec]
        predicted_concept = dict_concepts[class_num] 
        return predicted_concept


def get_shorthand_concept_C4(predicted_explanationVec):

    if(predicted_explanationVec == "a generic move not tied to a strategy"):
        predicted_explanation_to_concept = "null"
    elif(predicted_explanationVec == "blocks an opponent win and provides center dominance"):
        predicted_explanation_to_concept = "BW_CD"
    elif(predicted_explanationVec == "results in a three in a row and provides center dominance"):
        predicted_explanation_to_concept = "3IR_CD"
    elif(predicted_explanationVec == "blocks an opponent win and results in a three in a row"):
        predicted_explanation_to_concept = "BW_3IR"
    elif(predicted_explanationVec == "results in a three in a row but is blocked from a win"):
        predicted_explanation_to_concept = "3IR_Blocked"
    elif(predicted_explanationVec == "leads to a three in a row"):
        predicted_explanation_to_concept = "3IR"
    elif(predicted_explanationVec == "provides center dominance"):
        predicted_explanation_to_concept = "CD"
    elif(predicted_explanationVec == "blocks an opponent win"):
        predicted_explanation_to_concept = "BW"
    elif(predicted_explanationVec == "leads to a win"):
        predicted_explanation_to_concept = "W"

    return predicted_explanation_to_concept

dict_concepts_explanations = {
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves side fuel usage":0, 
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves main fuel usage":1,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander":2,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel usage":3,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground":4,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves main fuel":5,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel":6,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves side fuel":7,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground":8,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves side fuel":9,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground, conserves side fuel":10,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground":11,
    "encourages lander to land":12
}

dict_concepts = {
    0:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_lower_main_fuel_usage",
    1:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_lower_side_fuel_usage",
    2:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt",
    3:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact-encourages_lower_side_fuel_usage",
    4:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact",
    5:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_right_leg_contact-encourages_lower_side_fuel_usage",
    6:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact-encourages_right_leg_contact-encourages_lower_side_fuel_usage",
    7:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact-encourages_lower_main_fuel_usage",
    8: "encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_right_leg_contact",
    9:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_right_leg_contact-encourages_lower_main_fuel_usage",10:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact-encourages_right_leg_contact-encourages_lower_main_fuel_usage",
    11:"encourages_position_to_lander-encourages_slower_velocity-encourages_decrease_of_tilt-encourages_left_leg_contact-encourages_right_leg_contact",
    12: "encourages_to_land"
}