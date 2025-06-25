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

## Dataset
You can download the dataset from Dropbox and place it in the `data/` directory of the project:
- [Click here to download data.zip](https://www.dropbox.com/scl/fi/qiitjf0mwsx82bes4v39r/data.zip?rlkey=wn3hwvvoz2jhwmos66um3j02r&st=xrohlo0k&dl=0)

## Run
You can run the following commands directly in this folder. 
```sh
python main.py --dataset AMiner
python main.py --dataset ACM
python main.py --dataset DBLP
python main.py --dataset IMDB
```
