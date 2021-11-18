# HRVN: Fake News Differs From the Truth in Writing Style Detecting Fake News by Hierarchical Recursive Neural Networks

The implementation of HRVN in the paper: Fake News Differs From the Truth in Writing Style Detecting Fake News by Hierarchical Recursive Neural Networks

## Require
PyTorch >= 1.9.1

nltk >= 3.6.3

## About input data
1. We use the dataset ReCOVery as an example in the folder **ReCOVery** which has been split as training set, validation set and test set. For more detail about the ReCOVery dataset, you can refer to the webstite [ReCOVery](https://github.com/apurvamulay/ReCOVery)

2. We use **Stanford's GloVe 100d word embeddings** as word embedding in this paper, which is named as **glove.6B.100d.txt** in our code. The file of word embeddings can be downloaded from the webstite, [Glove.6b.100d](https://nlp.stanford.edu/projects/glove/).

3. For processing RST and CFT of ReCOVery dataset (or other news dataset), we use the code from the following website [Generate RST and CFG tree](https://github.com/jiyfeng/DPLP).

4. Here we provide a simple example of the format of RST and CFG in the folder **/data/strtree_RST** and **/data/strtree_CFG**. We generate RST and CFG tree for the example news **Original_text_news_1.txt**, which finally produces news_1.txt (RST) and news_1.txt (CFG) in the folders **/data/strtree_RST** and **/data/strtree_CFG** respectively.

## Reproducing Results
1. When getting all the input data including words embedding file **glove.6B.100d.txt**, folder **/data/strtree_RST** and **/data/strtree_CFG**, we can use the following commnads,


            python train.py
            
2. We can use pruning inside our method to decide width and depth by defining different max_depth and max_child, which is shown as the following commands,


           python train.py --max_depth==50 --max_child==10
