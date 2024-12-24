# Reddit Depression Detection

## Project Objective
The objective of this project is to train a random forest classifier to predict symptoms of depression from real Reddit text data. We use two methods to create linguistic features: fitting LDA and generating embeddings with (Distil)RoBERTa. To achieve this, we reimplement most parts of the paper [*Detecting Symptoms of Depression on Reddit*](https://dl.acm.org/doi/pdf/10.1145/3578503.3583621), including dataset generation and preprocessing.



## Dataset Generation
### control dataset
-  We create the control dataset by collecting non-mental health posts from authors at least 180 days before their first post in a depression-related subreddit.
### symptom dataset
For each system, we create dataset by collecting the posts from their respective subreddits. as shown in the table below.


## Preprocessing Plus Extracting Features
Two ways we extracted the features
1. LDA: 
We first tokenize the entire dataset using the 

The LDA Reddit topics portion of the paper uses an additional preprocessing step, i.e. removing top 100 most frequent words from the dataset.
The paper uses happierfuntokenizer, use happiestfuntokenizing · PyPI instead.

The paper uses MALLET for the LDA implementation. We use instead Gensim’s LdaMulticore model, as it will train much faster than SKLearn’s implementation.

2. DistRoberta: 
In the paper, the authors used the regular RoBERTa model with 12 transformer blocks and extracted the contextual embeddings from 10th layer for the downstream classification task. For this project, we use the DistilRoBERTa model, a distilled version of RoBERTa with only 6 transformer blocks (which makes it run faster), and we extract embeddings from the 5th layer for downstream classification. 

# Evaluation
You are only running a singular symptom vs control (the paper also does symptom vs control+all other symptoms).
E.g. Anger vs control, NOT anger vs control and anxiety and …



## Ethical consideration






