## INSTALLATION

## 1. Setup Conda

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```


<br>

## 2. Setup Python environment for CPU

```
# Clone GitHub repo
git clone https://github.com/vijaydwivedi75/emotion-gnn.git
cd emotion-gnn

# Install python environment
conda env create -f env_cpu.yml   

# Activate environment
conda activate emotion_gnn
```

Then install Pytorch Geometric using the following commannds.

```
python -m pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html  
python -m pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html  
python -m pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html  
python -m pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html  
python -m pip install torch-geometric  
```

## 3. Reproduce results of the DialogueGCN paper in terminal 

```
python pytorchgeometric/train.py --graph-model --no-cuda
```

## 4. Run specific 4-layer GNN model on the newly constructed graphs

Note: For now, sequential encoding is not being used
```
python main_emotion_node_classification.py --config configs/emotion_node_classification_MLP_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_MLP_GATED_IEMOCAP.json   
python main_emotion_node_classification.py --config configs/emotion_node_classification_GCN_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_GAT_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_GraphSage_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_GIN_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_MoNet_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_GatedGCN_IEMOCAP.json  
python main_emotion_node_classification.py --config configs/emotion_node_classification_GatedGCN_E_IEMOCAP.json  

```