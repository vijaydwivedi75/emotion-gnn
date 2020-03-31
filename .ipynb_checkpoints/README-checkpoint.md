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
conda install git
git clone xxx
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