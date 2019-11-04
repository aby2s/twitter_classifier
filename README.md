# Twitter author classifier
## Contents
* requirements.txt - Python requirements file
* author_classifier.py - Python script, which contains the final classifier.
* author_classifier.ipynb - Jupyter notebook which contains some of my thoughts about the task and steps that I took before I came to the final solution
* data - Provided data with the modified train set (see [Train set modifications](#train-set-modifications))
## Requirements
I run this code on Python 3.6. Other Python 3.x versions could be suitable too. Requirements are fixed in a requirements file. To install them run:
```console
$ pip install -r requirements.txt
```
You should also download SpaCy models and data for the English language:
```console
$ python -m spacy download en_core_web_sm
```
## How to run author_classifier script
To train and perform validation using a part of the train set:
```console
$ python author_classifier.py --task fit --model_path ./model1 --ds_path ./data/train_set.csv --test_size 0.1
```
To train using the whole train set:
```console
$ python author_classifier.py --task fit --model_path ./model_full --ds_path ./data/train_set.csv 
```
To perform k-fold validation:
```console
$ python author_classifier.py --task kfold --model_path ./model1 --ds_path ./data/train_set.csv --nsplits 5
```
To predict on the test set:
```console
$ python author_classifier.py --task evaluation --model_path ./model1 --ds_path ./data/test_set.csv --eval_path ./predictions.csv
```
## Train set modifications
I've made several modifications to train_set.csv:
* Added quotes to tweets on lines 3678 and 7850
* Changed value of a column minutes on line 3674 ellen from 20f to 20
I didn't find any problems with test_set.csv.
## CatBoost and a GPU 
CatBoost is set to train on a GPU. If you don't have a GPU with 6GB VRAM available, you can run training on a CPU. To do so, change task_type parameter to CPU in CatBoost constructor. It will take more time to train (a couple of hours on an average CPU vs. several minutes on a GPU), but it will work fine. CatBoost doesn't require a GPU on inference.
