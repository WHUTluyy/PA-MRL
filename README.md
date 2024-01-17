# PA-MRL

## Environment Setup

python==3.7.13    
scipy==1.7.3
numpy==1.20.0
pandas==1.3.5
networkx==2.6.3  
pytorch==1.13.1          
rdkit==2023.3.2                           
torch-geometric==1.7.0                    
torch-scatter==2.1.1                     
tqdm==4.66.1    

                
## Training

You can pretrain the model by

```
python pretrain.py
```

## Evaluation

You can evaluate the pretrained model by finetuning on downstream tasks

Download the downstream data from https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function, and save the .csv files in the ./finetune/dataset/[dataset_name]/raw/, where [dataset_name] is replaced by the downstream dataset name.
For example, toxcast.csv is saved in './finetune/dataset/toxcast/raw/toxcast.csv'.

```
cd finetune
mkdir model_checkpoints
python finetune.py
```
