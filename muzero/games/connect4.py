import datetime
import os
import json
import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import spacy
import MEM_utils 

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1 # None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 6, 7)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(7))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 5  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.2
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 1  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 8  # Number of channels in reward head
        self.reduced_channels_value = 8  # Number of channels in value head
        self.reduced_channels_policy = 8  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs

        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-5  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None # None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
            

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Connect4()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.

        """
        observation, reward, done, bool_win, bool_draw, concept_arr, board, prev_board, predicted_concept = self.env.step(action)
        # print("------- in game.step")
        # print(board)
        # print(prev_board)
        return observation, reward * 10, done, bool_win, bool_draw, concept_arr, board, prev_board, predicted_concept

    def get_explanation(self, action):

        explanation, explanation_to_concept, gt_concept = self.env.get_explanation(action)
        return explanation, explanation_to_concept, gt_concept

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"


class Connect4:
    def __init__(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1

        self.blocking_reward = 0.5
        self.center_column_reward = 0.1
        self.ThreeinRowBlocked_reward = 0 
        self.ThreeinRow_reward = 0.1
        self.TwoInRow_reward = 0
        self.null_reward = 0
        self.win_reward = 1

        # #### FOR RUNNING WITH NO REWARD SHAPING (Emulating original sparse reward) #####
        # self.blocking_reward = 0
        # self.center_column_reward = 0
        # self.ThreeinRow_reward = 0
        # self.TwoInRow_reward = 0
        # self.null_reward = 0
        # self.win_reward = 1
        # self.ThreeinRowBlocked_reward = 0 

        ###### FOR RUNNING WITH NO MEM SET BOOL = FALSE
        self.mem_flag = True


        self.prev_board = numpy.zeros((6, 7), dtype="int32")

        self.count = 0
        with open('../MEM/vocab_C4.json', 'r') as openfile:
            self.vocab = json.load(openfile)
        
        self.MEM_model = torch.load("../MEM/test_C4/MEM-C4_seed_42.pth", map_location=torch.device('cpu'))
        self.MEM_model.eval()


    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1
        return self.get_observation()

    ###THIS FUNCTION IS USED IN self_play_C4.py
    def get_explanation(self, action):
        prev_board_holder = self.prev_board.copy()
        board_holder = self.board.copy()

        self.prev_board = self.board[::-1].copy()
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                row = i
                col = action
                break

        self.current_board = self.board[::-1].copy()

        _, gt_concept = self.reward_functionFromStates(row, col) #get what the groundtruth concept should be
        player_ = self.player*-1
        predicted_explanationVec, predicted_explanation_to_concept, _ = self.getConcept_MEM(player_)

        self.prev_board = prev_board_holder
        self.board = board_holder

        return predicted_explanationVec, predicted_explanation_to_concept, gt_concept

    def step(self, action):
        #row and column 
        row = 0 
        col = 0
        self.count+=1

        self.prev_board = self.board[::-1].copy()
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                row = i
                col = action
                break

        self.current_board = self.board[::-1].copy()
        bool_win =  self.have_winner()
        done = bool_win or len(self.legal_actions()) == 0
        bool_draw = not bool_win and len(self.legal_actions()) == 0
        
        _, gt_concept = self.reward_functionFromStates(row, col) #get what the groundtruth concept should be

        player_ = self.player*-1


        if(self.mem_flag):

            predicted_explanationVec, predicted_explanation_to_concept, _ = self.getConcept_MEM(player_)
            predicted_concept = predicted_explanation_to_concept
            reward = self.getRewards(predicted_concept)
        else:
            predicted_concept = ""
            reward = self.getRewards(gt_concept) #g_concept or predicted_concept

        self.player *= -1

        return self.get_observation(), reward, done, bool_win,bool_draw, gt_concept, self.current_board, self.prev_board, predicted_concept


    def getRewards(self, predicted_concept):
        reward = 0
        if(predicted_concept == "null" or predicted_concept == "3IR_Blocked"):
            reward = self.null_reward
        elif(predicted_concept == "CD"):
            reward = self.center_column_reward
        elif(predicted_concept == "3IR" or predicted_concept == "3IR_CD"):
            reward = self.ThreeinRow_reward
        elif(predicted_concept == "W"):
            reward = self.win_reward
        elif(predicted_concept == "BW" or predicted_concept == "BW_3IR" or predicted_concept=="BW_CD"):
            reward = self.blocking_reward
        return reward


    def getConcept_MEM(self, player_):
        
        #convert to tensor and flatten the prevBoard and current board
        copy_prev_board = self.prev_board.copy()
        copy_curr_board = self.board[::-1].copy()    
        x_prevBoard= torch.Tensor(copy_prev_board.flatten()).reshape((1,42))
        x_currBoard = torch.Tensor(copy_curr_board.flatten()).reshape((1,42))
        #add x_gameOver, x_player 
        x_gameOver = []
        x_player = []
        if(self.have_winner() or len(self.legal_actions()) == 0):
            x_gameOver.append(1)      
        else:
            x_gameOver.append(0)
        x_player.append(0 if player_ == 1 else 1)
        x_gameOver = torch.unsqueeze(torch.Tensor(x_gameOver),1)
        x_player = torch.unsqueeze(torch.Tensor(x_player),1)

        ### PERFORM PADDING ######
        ### for CNN add row of 0's at the top to make 7x7, reshape to 1, 49
        x_prevBoard = F.pad(x_prevBoard, (0, 49-x_prevBoard.size(1)),"constant", 0)
        x_currBoard = F.pad(x_currBoard, (0, 49-x_currBoard.size(1)),"constant", 0)

        x_gameOver= F.pad(x_gameOver, (0, 49-x_gameOver.size(1)),"constant", 99)
        x_player = F.pad(x_player, (0, 49-x_player.size(1)),"constant", 99)
        ##########################
        maxLength = 11 

        l2_norm_arr = []
        for j, j_explanation in enumerate(MEM_utils.get_explanation_list("C4")):
            get_encoded_j_explanation = MEM_utils.encoding(MEM_utils.tokenize_explanations([j_explanation]), self.vocab, maxLength)
            new_explanation_x = torch.tensor(get_encoded_j_explanation.copy())
            mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation = MEM_utils.get_masks_C4(x_prevBoard,x_currBoard, x_gameOver, x_player, new_explanation_x)

            with torch.no_grad():
                state_embed, explanation_embed = self.MEM_model.forward(x_prevBoard, x_currBoard, x_gameOver, x_player, new_explanation_x, mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation)
            difference = state_embed - explanation_embed
            l2_norm = torch.linalg.norm(difference, dim=1, ord=2)
            l2_norm_arr.append(l2_norm.cpu())

        predicted_index = numpy.argmin(numpy.array(l2_norm_arr))
        predicted_explanationVec = MEM_utils.get_explanation_list("C4")[predicted_index]
        predicted_explanation_to_concept = MEM_utils.get_shorthand_concept_C4(predicted_explanationVec)
      
        return predicted_explanationVec, predicted_explanation_to_concept, l2_norm

    def reward_functionFromStates(self, row, col):
        #this function first inspects whether (1)certain concepts exist on the current game board,
        # (2) then generates a concept_list accordingly, and provides what the reward for the state should be accordingly
        # (3) then assigns what the gt_class should be based on concept_list

        ######### finding the concepts
        concept_list = []
        r_block, _, concept_block = self.reward_blocking(row, col)
        r_three, _, concept_three = self.reward_buildThreeinRow(row, col)
        r_three_blocked, _, concept_three_blocked = self.reward_buildThreeinRowBlocked(row, col)
        r_center, _, concept_center = self.center_column(row,col)
        reward = 0

        if self.have_winner():
            concept_list.append("W")
        if(concept_block != ""):
            concept_list.append("BW")
        if(concept_three != ""):
            concept_list.append("3IR")
        if(concept_center != ""):
            concept_list.append("CD")
        if(concept_three_blocked != ""):
            concept_list.append("3IR_Blocked")
            
        #### giving reward  based on concept list
        if("W" in concept_list):
            reward = 1
        elif(not self.have_winner() and len(self.legal_actions()) == 0):
            reward = 0 #draw reward
        elif("3IR_Blocked" in concept_list):
            reward = 0
        elif("BW" in concept_list or "BW_CD" in concept_list or "BW_3IR" in concept_list):
            reward = self.blocking_reward
        elif("3IR" in concept_list or "3IR_CD" in concept_list):
            reward = self.ThreeinRow_reward
        elif("CD" in concept_list):
            reward = self.center_column_reward
        else:
            reward = 0
        
        # assigning the GT class based on concept_list
        if("W" in concept_list):
            gt_class = "W"
        elif("CD" in concept_list and "BW" in concept_list):
            gt_class = "BW_CD"
        elif("CD" in concept_list and "3IR" in concept_list):
            gt_class = "3IR_CD"
        elif("BW" in concept_list and "3IR" in concept_list):
            gt_class = "BW_3IR"
        elif("3IR_Blocked" in concept_list):
            gt_class = "3IR_Blocked"
        elif("3IR" in concept_list):
            gt_class = "3IR"
        elif("CD" in concept_list):
            gt_class = "CD"
        elif("BW" in concept_list):
            gt_class = "BW"
        else:
            gt_class = "null"
        
        return reward, gt_class

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((6, 7), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def reward_buildThreeinRowBlocked(self,row,col):
        reward = 0
        concept_blocked = ""
        bool_check_blocked = False
        loc = 0

        #horizontal checks
        if(col >= 2):
            if(self.board[row][col-1]==self.player and self.board[row][col-2] == self.player):
                if (col == 2):
                    if(self.board[row][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                elif (col == 6):
                    if(self.board[row][col-3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True 
                elif (col >= 3 and col <= 5):
                    if(self.board[row][col-3] == self.player*-1 and self.board[row][col+1] == self.player*-1):
                        bool_check_blocked = True

        if(col <= 4):
            if(self.board[row][col+1] == self.player and self.board[row][col+2] == self.player):
                if (col == 0):
                    if(self.board[row][col+3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                elif (col == 4): 
                    if(self.board[row][col-1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                elif (col >= 1 and col <= 3): 
                    if(self.board[row][col-1] == self.player*-1 and self.board[row][col+3] ==self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
        
        if(col >= 1 and col <= 5):
            if(self.board[row][col+1] == self.player and self.board[row][col-1] == self.player):
                if (col == 1):
                    if(self.board[row][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                elif (col == 5):
                    if(self.board[row][col-2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                elif (col >= 2 and col <= 4):
                    if(self.board[row][col-2] == self.player*-1 and self.board[row][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
        
        # No vertical checks neededd

        #positive diagnonal checks 
        if(row >= 2 and col >= 2):
            if(self.board[row-1][col-1] == self.player and self.board[row-2][col-2] == self.player):
                if((row == 2 and col == 6) or (row == 5 and col == 2)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 5 and col >= 3):
                    if(self.board[row-3][col-3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 2 and col <= 5):
                    if(self.board[row+1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 2 and row <= 4):
                    if(self.board[row+1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 6 and row >= 3):
                    if(self.board[row-3][col-3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row <=4 and col <=5) and (row >= 3 and col >=3)):
                    if(self.board[row-3][col-3] == self.player*-1 and self.board[row+1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True

        if (row <= 3 and col <= 4):
            if(self.board[row+1][col+1] == self.player and self.board[row+2][col+2] == self.player):
                if ((col == 4 and row == 0) or (col == 0 and row == 3)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 3 and col >= 1):
                    if(self.board[row-1][col-1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 0 and col <= 3):
                    if(self.board[row+3][col+3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 0 and row <= 2):
                    if(self.board[row+3][col+3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 4 and row >= 1):
                    if(self.board[row-1][col-1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row >= 1 and col >= 1) and (row <= 2 and col <= 3)):
                    if (self.board[row-1][col-1] == self.player*-1 and self.board[row+3][col+3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True          
        
        if (row >= 1 and col >= 1 and row <= 4 and col <= 5):
            if(self.board[row-1][col-1] == self.player and self.board[row+1][col+1] == self.player):
                if((row == 1 and col == 5) or (row == 4 and col == 1)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 1 and col <= 4):
                    if(self.board[row+2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 4 and col >= 2):
                    if(self.board[row-2][col-2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 1 and row <= 3):
                    if(self.board[row+2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 5 and row >= 2):
                    if(self.board[row-2][col-2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row >= 2 and col >= 2) and (row <= 3 and col <= 4)):
                    if (self.board[row-2][col-2] == self.player*-1 and self.board[row+2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True          

        #negative diagonal checks 
        if (row >= 2 and col <= 4):
            if(self.board[row-1][col+1] == self.player and self.board[row-2][col+2] == self.player):
                if ((row == 2 and col == 0) or (row == 5 and col == 4)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 2 and col >= 1):
                    if(self.board[row+1][col-1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 5 and col <= 3):
                    if(self.board[row-3][col+3] ==self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 0 and row >= 3):
                    if(self.board[row-3][col+3] ==self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 4 and row <= 4):
                    if(self.board[row+1][col-1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row <= 4 and col >= 1) and (row >= 3 and col <= 3)):
                    if (self.board[row+1][col-1] == self.player*-1 and self.board[row-3][col+3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True

        if (col >= 2 and row <= 3):
            if(self.board[row+1][col-1] == self.player and self.board[row+2][col-2] == self.player):
                if((row == 0 and col == 2) or (row == 3 and col == 6)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 0 and col >= 3):
                    if(self.board[row+3][col-3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 3 and col <= 5):
                    if(self.board[row-1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 6 and row <= 2):
                    if(self.board[row+3][col-3] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 2 and row >= 1):
                    if(self.board[row-1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row >= 1 and col <= 5) and (row <= 2 and col >= 3)):
                    if (self.board[row+3][col-3] == self.player*-1 and self.board[row-1][col+1] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                
        if (row <= 4 and col >= 1 and row >= 1 and col <= 5):
            if(self.board[row+1][col-1] == self.player and self.board[row-1][col+1] == self.player):
                if((row == 1 and col == 1) or (row == 4 and col == 5)): # Special case to handle 3IR in the corner of the board
                    reward = self.ThreeinRowBlocked_reward
                    bool_check_blocked = True
                if (row == 1 and col >= 2):
                    if(self.board[row+2][col-2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (row == 4 and col <= 4):
                    if(self.board[row-2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 5 and row <= 3):
                    if(self.board[row+2][col-2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if (col == 1 and row >= 2):
                    if(self.board[row-2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                if ((row <= 3 and col >= 2) and (row >= 2 and col <= 4)):
                    if (self.board[row+2][col-2] == self.player*-1 and self.board[row-2][col+2] == self.player*-1):
                        reward = self.ThreeinRowBlocked_reward
                        bool_check_blocked = True
                
        if(bool_check_blocked):
            concept_blocked = "3IR_Blocked"
        
        return reward, bool_check_blocked, concept_blocked 
    
    def reward_blocking(self,row, col):
        #horizontal checks
        concept = ""
        reward = 0
        bool_check = False
        if(col >= 2 and col <=5):
            if(self.board[row][col-2] == self.player*-1 and self.board[row][col-1] == self.player*-1 and self.board[row][col+1] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col >= 3):
            if(self.board[row][col-3] == self.player*-1 and self.board[row][col-2]==self.player*-1 and self.board[row][col-1] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col<= 3):
            if(self.board[row][col+1] == self.player*-1 and self.board[row][col+2] == self.player*-1 and self.board[row][col+3] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col>=1 and col <=4):
            if(self.board[row][col-1] == self.player*-1 and self.board[row][col+1] == self.player*-1 and self.board[row][col+2] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        #vertical checks 
        if(row >=3):
            if(self.board[row-1][col] == self.player*-1 and self.board[row-2][col] == self.player*-1 and self.board[row-3][col] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True

        #positive diagonal checks
        if(col <=6 and col>=3 and row <=5 and row >=3):
            if(self.board[row-1][col-1] == self.player*-1 and self.board[row-2][col-2] == self.player*-1 and self.board[row-3][col-3] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col >=1 and col <=4 and row >=1 and row <=3):
            if(self.board[row-1][col-1] == self.player*-1 and self.board[row+1][col+1] == self.player*-1 and self.board[row+2][col+2] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col>=2 and col <=5 and row>=2 and row <=4):
            if(self.board[row+1][col+1] == self.player*-1 and self.board[row-1][col-1] == self.player*-1 and self.board[row-2][col-2] == self.player*-1):
                reward = self.blocking_reward 
                bool_check = True
        if(col>=0 and col <=3 and row>=0 and row <=2):
            if(self.board[row+1][col+1] == self.player*-1 and self.board[row+2][col+2] == self.player*-1 and self.board[row+3][col+3] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        #negative diagonal check
        if(col <=6 and col >=3 and row >=0 and row <=2):
            if(self.board[row+1][col-1] == self.player*-1 and self.board[row+2][col-2] == self.player*-1 and self.board[row+3][col-3] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(col >=0 and col <=3 and row <=5 and row >=3):
            if(self.board[row-1][col+1] == self.player*-1 and self.board[row-2][col+2] == self.player*-1 and self.board[row-3][col+3] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(row<=4 and row >=2 and col>=1 and col <=4):
            if(self.board[row+1][col-1] == self.player*-1 and self.board[row-1][col+1] == self.player*-1 and self.board[row-2][col+2] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(row<=3 and row >=1 and col >=2 and col <=5):
            if(self.board[row-1][col+1] == self.player*-1 and self.board[row+1][col-1] == self.player*-1 and self.board[row+2][col-2] == self.player*-1):
                reward = self.blocking_reward
                bool_check = True
        if(bool_check == True):
            concept = "BW"
        return reward, bool_check, concept

    def center_column(self, row, col):
        reward = 0
        bool_check = False
        concept = ""
        if(col == 3):
            if(self.board[row][col] == self.player):
                reward = self.center_column_reward
                bool_check = True 

        if(bool_check):
            concept = "CD"
        return reward, bool_check, concept  

    def reward_buildThreeinRow(self,row,col):
        reward = 0
        concept = ""
        bool_check = False
        loc = 0
        #horizontal checks
        if(col >= 2):
            if(self.board[row][col-1]==self.player and self.board[row][col-2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        if(col <= 4):
            if(self.board[row][col+1] == self.player and self.board[row][col+2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True
        
        if(col >= 1 and col <= 5):
            if(self.board[row][col+1] == self.player and self.board[row][col-1] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True
        
        #vertical checks
        if(row >= 2 and row <= 4):
            if(self.board[row-1][col] == self.player and self.board[row-2][col] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        #positive diagnonal checks 
        if(row >= 2 and col >= 2):
            if(self.board[row-1][col-1] == self.player and self.board[row-2][col-2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True
        
        if (row <= 3 and col <= 4):
            if(self.board[row+1][col+1] == self.player and self.board[row+2][col+2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True
        
        if (row >= 1 and col >= 1 and row <= 4 and col <= 5):
            if(self.board[row-1][col-1] == self.player and self.board[row+1][col+1] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        #negative diagonal checks 
        if (row >= 2 and col <= 4):
            if(self.board[row-1][col+1] == self.player and self.board[row-2][col+2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        if (col >= 2 and row <= 3):
            if(self.board[row+1][col-1] == self.player and self.board[row+2][col-2] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        if (row <= 4 and col >= 1 and row >= 1 and col <= 5):
            if(self.board[row+1][col-1] == self.player and self.board[row-1][col+1] == self.player):
                reward = self.ThreeinRow_reward
                bool_check = True

        if(bool_check):
            concept = "3IR"
        
        return reward, bool_check, concept 

    def have_winner(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j][i + 1] == self.player
                    and self.board[j][i + 2] == self.player
                    and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i] == self.player
                    and self.board[j + 2][i] == self.player
                    and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i + 1] == self.player
                    and self.board[j + 2][i + 2] == self.player
                    and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j - 1][i + 1] == self.player
                    and self.board[j - 2][i + 2] == self.player
                    and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        for k in range(3):
            for l in range(4):
                sub_board = board[k : k + 4, l : l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = numpy.where(sub_board[i, :] == 0)[0][0]
                        if numpy.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if self.player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if self.player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = numpy.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = numpy.where(diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if self.player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = numpy.where(anti_diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if self.player * sum(anti_diag) > 0:
                            return action

        return action

    def render(self):
        print(self.board[::-1])
