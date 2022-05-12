# Neural Coreference Resolution for Arabic

## Introduction
This repository contains code introduced in the following paper:
 
**[Neural Coreference Resolution for Arabic](https://www.aclweb.org/anthology/2020.crac-1.11/)**  
Abdulrahman Aloraini*, Juntao Yu* and Massimo Poesio 	`*equal contribution`  
In *Proceedings of the Third Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC@COLING)*, 2020.

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that run `setup.sh` to download the fastText embeddings that required by the system and compile the Tensorflow custom kernels.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://essexuniversity.box.com/s/4ors9kax5zfn8r1q63uetftyugasn096). We provide two pre-trained models:
   * One (arabic_cleaned_arabert) for Lee et al (2018) style training.
   * The second model (arabic_cleaned_arabert_e2e_annealing) that uses the predicted mention output from the [Yu et al (2020)](https://github.com/juntaoy/dali-md) and also the best model from our paper.
   * We include the predicted mentions used in our evaluation for all three datasets (train, dev and test sets).
   * In the folder you will also find a file called *char_vocab.arabic.txt* which is the vocabulary file for character-based embeddings used by our pre-trained models.
* Put the downloaded models along with the *char_vocab.arabic.txt* in the root folder of the code.
* Modifiy the *test_path* and *conll_test_path* accordingly:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file must in the following format:
   
   ```
  {
  "clusters": [[[0,0],[5,5]],[[2,3],[7,8]],
  "pred_mentions":[[0,0],[2,3],[5,5],[7,9]], #Optional
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["He", "washed", "the", "car", "yesteday","."],["Really","?","it", "was", "raining","yesteday","!"]],
  "speakers": [["sp1", "sp1", "sp1", "sp1", "sp1"], ["sp1", "sp1", "sp1", "sp1", "sp1","sp1"],["sp2","sp2","sp2","sp2","sp2","sp2","sp2"]]
  }
  ```
  
  * For "clusters" and "pred_mentions" the mentions contain two properties \[start_index, end_index\] the indices are counted in document level and both inclusive.
  * the *conll_test_path* is the path to the file of gold data in CoNLL format, see the [CoNLL 2012 shared task page](http://conll.cemantix.org/2012/introduction.html) for more detail
  * For how to create the json and CoNLL files please follow the instractions from the [Lee et al (2018)](https://github.com/kentonl/e2e-coref).
  * You can preprocess the Arabic tokens by using `python preprocess_arabic.py test.jsonlines test.cleaned.jsonlines`.
      
* Then you need to run the `extract_bert_features.sh` to compute the BERT embeddings for the test set.
* Then use `python evaluate.py config_name` to start your evaluation.

## To train your own model
* To train your own model you need first create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you need to run the `extract_bert_features.sh` to compute the BERT embeddings for training, development and test sets.
* Finally you can start training by using `python train.py config_name`

## Training speed
The cluster ranking model takes about 40 hours to train (400k steps) on a GTX 1080Ti GPU. 
