# Graph-based dependency parsing using BiLSTM feature extractors

A PyTorch implementation of the BIST graph-based parser as described 
in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198).

The parser acheives 94.38% UAS and 92.93% LAS on the standard Penn Treebank dataset (Standford Dependencies). 

## Requirements
* Python 3.7
* PyTorch 1.8.1
* NLTK 3.6.2

## Data format
The software requires having a `train.conll` and `dev.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat).

### Train a parsing model
For training a graph-based parsing model, run:
```
python main.py --train_path data/train.conll --dev_path data/dev.conll --epochs 30 --lr 1e-3 --w_emb_dim 100 --pos_emb_dim 25 --lstm_hid_dim 125 --mlp_hid_dim 100 --n_lstm_layers 2
```
### Parse data
For parsing data with a previously trained model, run:
```
python main.py --test_path data/test.conll --model_dir [model directory] --do_eval
```
