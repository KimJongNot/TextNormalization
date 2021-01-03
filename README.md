# TextNormalization

## Indonesian Text Normalization
The purpose of this project is to help normalise indonesian text especially texts written on Twitter. The focus is Indonesian language not included special dialects. In this project we use Word Embedding to capture relationships between words. The model trained are Word2Vec and FastText models. 

### Method Used
* Word Embedding (Word2Vec and FastText)
* Text Similarities (Jaro-winkler and Leveinshten)

### Technologies
* Python
* Pandas, Jupyter
* Gensim

### Overview
The data used are collected from Twitter which tweets written in 2018. Other data is KBBI (Kamus Besar Bahasa Indonesia) from kbbi kemdikbud. The goal is to use Word Embedding model to normalize text. What expected is to normalize 1 non standard word to be in standard form. For example:
* buanyak -> banyak
* ilang -> hilang
* maem -> makan


### Additional notes
* The word2Vec model is better to normalize frequent words and is part of the corpus.
* The FastText model is better to normalize unfrequent words or or words that can not be found in corpus.
* The model is not in this repo. Please contact me to download the model