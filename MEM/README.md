# MEM in S2E
This readme provides information on how to train and evaluate the joint embedding models (MEMs) used in the S2E framework.
All instructions assume Python 3.9.7 and an Ubuntu Operating System

## Training
To train the joint embedding models used in S2E for the Connect 4 or Lunar Lander domain, run the commands below:
```
python trainer_C4.py --filename
python trainer_LL.py --filename
```
> Hyperparameters used in the paper are set at the bottom of each respective trainer file.
> Model architectures are povided in ``models.py``
> Datasets are provided in ``datasets.zip``; ``data_LL.py`` and ``data_C4.py`` show dataloading for training each model

## Evaluation
To evaluate the joint embedding models, use the commands below:
```
python test_model_C4.py --tsne True --cm True
python test_model_LL.py --tsne True --cm True
```
> The ``--tsne`` and ``cm`` parameters allow for generation of TNSE plots on the test set and Confusion Matrix analysis for Recall@1. 
> Additionally, with the above commands, Recall@k=1,2,3 are printed to the terminal. 

## Pre-Trained Models 
Pre-Trained model provided in ``test_C4`` and ``test_LL`` folders.  

> You can run model evaluation on the existing models in these folders.
