# HRVN: Fake News Differs From the Truth in Writing Style Detecting Fake News by Hierarchical Recursive Neural Networks

The implementation of HRVB in our paper: Fake News Differs From the Truth in Writing Style Detecting Fake News by Hierarchical Recursive Neural Networks

## Require
PyTorch >= 1.9.1

nltk >= 3.6.3

## About input data
1. We use the dataset ReCOVery as an example in the folder **ReCOVery** which has been split as training set, validation set and test set. For more detail about the ReCOVery dataset, you can refer to the webstite [ReCOVery](https://github.com/apurvamulay/ReCOVery)

2. We use **Stanford's GloVe 100d word embeddings** as word embedding in this paper, which is named as **glove.6B.100d.txt** in our code. The file of word embeddings can be downloaded from the webstite, [Embedding of words using Glove 100d](https://pages.github.com/).

3. For processing RST and CFT of ReCOVery dataset (or other news dataset), we use the code from the following website [Code for RST_and_CFG](https://pages.github.com/). Here we provide a simple example of the format of RST and CFG in the folder **/data/strtree_RST** and **/data/strtree_CFG**.

## Reproducing Results
When preparing all the input data including words embedding file **glove.6B.100d.txt**, folder **/data/strtree_RST** and **/data/strtree_CFG**, we can use the following commnads


            python train.py
