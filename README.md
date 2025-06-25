## Addressing Graph Heterogeneity and Heterophily from A Spectral Perspective (H2SGNN)
This code contains a PyTorch implementation of "Addressing Graph Heterogeneity and Heterophily from ASpectral Perspective" (H2SGNN)
## Environment Settings
- pytorch 1.12.1
- numpy 1.23.1
- dgl 0.9.1
- torch-geometric 2.1.0
- tqdm 4.64.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.6
  
The dataset is already in the data folder. You can run the following commands directly in this folder. 
```sh
python main.py --dataset AMiner
python main.py --dataset ACM
python main.py --dataset DBLP
python main.py --dataset IMDB
```
