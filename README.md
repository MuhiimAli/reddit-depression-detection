# reddit-depression-detection

# Generating the dataset
## control dataset
- of the same users, consisting of posts from non-depression-related subreddits.
- "only [keep] non-mental health posts by authors that were at least 180 days older than their index (earliest) post in a mental health subreddit"

##
Two ways we extracted the features
In the paper, the authors used the regular RoBERTa model with 12 transformer blocks and extracted the contextual embeddings from 10th layer for the downstream classification task. For this project, we use the DistilRoBERTa model, a distilled version of RoBERTa with only 6 transformer blocks (which makes it run faster), and we extract embeddings from the 5th layer for downstream classification. 

"Each word (token, to be specific) has a single embedding representation. To get the embedding for a post, average together the embeddings of each token!"


