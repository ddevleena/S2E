# Utilizing MEM for RL agent benefits in S2E
This readme provides information on to leverage the trained joint embedding models in ``/MEM`` to benefit RL training. 
Note, we leverage a state-of-the-art RL algoritm Muzero as the RL agent. 
**All code code in this folder is adapted from the open source [Muzero](https://github.com/werner-duvaud/muzero-general).** 

All instructions assume Python 3.9.7 and an Ubuntu Operating System

## Training 
To train Muzero with or without using our joint embedding models to inform reward shaping, run the following commands:
```
python muzero.py
```
> Select the appropriate domain (Lunar Lander or Connect4). Then select ``0. Train``.
### Option for leveraging joint embedding model 
If you would like to utilize the trained joint embedding model to inform reward shaping, in ``games/lunarlander.py`` or ``games/connect4.py``, set ``self.mem_flag= True``, 
otherwise set ``self.mem_flag=False``.

## Evaluation 
1) To monitor the agent's training, utilize tensorboard via the following command: 
```
tensorboard --logdir ./results
```
2) You may also render some self-play games or play against MuZero. To do this, run: 
``` python muzero.py ```
> Select the appropriate domain (Lunar Lander or Connect 4). Then select ``1. Load pretrained model ``. Then select either ``2. Render some self-play games`` or ``3. Play against Muzero``.

## Pre-Trained Models
Existing Pre-Trained RL agent checkpoints are provided in the ``results/`` folder. 
We provide a model checkpoint for each of the study conditions used to evaluate efficacy of our joint embedding models in informing reward shaping. 
The performances of these models can be visualized using tensorboard or the other evaluation options above.

For Connect 4, we provide:
1) Baseline_GT_RS - Agent trained with w/expert-defined reward shaping (uppder bound for our joint embedding model)
2) Baseline_No_RS - Agent trained with w/the current existing no reward shaping 
3) MEM_RS - Agent trained w/joint embedding model informed reward shaping 

For Lunar Lander, we provide: 
1) Baseline_GT_RS - Agent trained w/the current existing expert-defined reward shpaing 
2) MEM_RS - Agent trained w/joint embedding model informed reward shaping

