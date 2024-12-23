#imports
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import os
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer

# List of depression subreddits in the paper
depression_subreddits = ["Anger",
    "anhedonia", "DeadBedrooms",
    "Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack",
    "DecisionMaking", "shouldi",
    "bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous",
    "chronicfatigue", "Fatigue",
    "ForeverAlone", "lonely",
    "cry", "grief", "sad", "Sadness",
    "AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou",
    "insomnia", "sleep",
    "cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus",
    "AdultSelfHarm", "selfharm", "SuicideWatch",
    "Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"
]





def load():
  """Load pickles"""
  filepath = "/users/mali37/student.pkl"
  file = open(filepath,'rb')
  df = pd.read_pickle(file)
  return df
 



def dataset_generation(use_cache=True):
  """Build control and symptom datasets with improved performance"""
  if use_cache and os.path.exists('anxiety_dataset.pkl') and os.path.exists('control_dataset.pkl') and os.path.exists('depression_dataset.pkl'):
      print("Loading from cache...")
      with open('anxiety_dataset.pkl', 'rb') as f:
          anxiety_dataset = pickle.load(f)
      with open('depression_dataset.pkl', 'rb') as f:
          depression_posts = pickle.load(f)
      with open('control_dataset.pkl', 'rb') as f:
          control_dataset = pickle.load(f)
      print(f"Anxiety dataset size: {len(anxiety_dataset)}")
      print(f"depression dataset size: {len(depression_posts)}")
      print(f"Control dataset size: {len(control_dataset)}")
      print(f"Number of unique users in control: {len(control_dataset['author'].unique())}")
      return anxiety_dataset, control_dataset, depression_posts
  
  df = load()
  
  # Create anxiety dataset (unchanged)
  anxiety_dataset = df[df['subreddit'].isin([
      'anxiety', 'AnxietyDepression', 'HealthAnxiety', 'PanicAttack'
  ])]
  
  depression_mask = df['subreddit'].isin(depression_subreddits)
  depression_posts = df[depression_mask]
  
  
  user_first_depression_post = (
      depression_posts
      .groupby('author')['created_utc']
      .min()
  )
  
  # Create control dataset
  df['first_depression_post'] = df['author'].map(user_first_depression_post)
  cutoff_dates = df['first_depression_post'] - (180 * 24 * 60 * 60)
  
  control_mask = (
      (~depression_mask) &
      (df['created_utc'] < cutoff_dates) &
      (~df['first_depression_post'].isna())
  )
  
  control_dataset = df[control_mask]
  
  # Save to cache
  if use_cache:
      print("Saving to cache...")
      with open('anxiety_dataset.pkl', 'wb') as f:
          pickle.dump(anxiety_dataset, f)
      with open('depression_dataset.pkl', 'wb') as f:
          pickle.dump(depression_posts, f)
      with open('control_dataset.pkl', 'wb') as f:
          pickle.dump(control_dataset, f)
  
  print(f"Anxiety dataset size: {len(anxiety_dataset)}")
  print(f"depression dataset size: {len(depression_posts)}")
  print(f"Control dataset size: {len(control_dataset)}")
  print(f"Number of unique users in control: {len(control_dataset['author'].unique())}")
  
  return anxiety_dataset, control_dataset, depression_posts





def tokenize_text(text):
    """Tokenize a single text using happiestfuntokenizing"""
    tokenizer = Tokenizer()
    if pd.isna(text):
        return []
    try:
        return tokenizer.tokenize(str(text))
    except:
        return []

def tokenize(dataset, use_cache=True, cache_prefix=''):
    """Tokenize dataset using happiestfuntokenizing"""
    cache_file = f'{cache_prefix}_tokenized.pkl'
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading tokenized {cache_prefix} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Tokenizing {cache_prefix} dataset...")
    
    # Create a copy to avoid modifying the original
    tokenized_df = dataset.copy()
    
    # Apply tokenization to 'text' column
    tokenized_df['tokenized_text'] = tokenized_df['text'].apply(tokenize_text)
    
    # Calculate token count
    tokenized_df['token_count'] = tokenized_df['tokenized_text'].apply(len)
    
    # Remove empty posts
    tokenized_df = tokenized_df[tokenized_df['token_count'] > 0]
    
    if use_cache:
        print(f"Saving tokenized {cache_prefix} to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_df, f)
    
    return tokenized_df



def stop_words():
  """Find top 100 words from Reddit dataset to use as stop words"""
  pass

if __name__ == "__main__":
  anxiety_df, control_df, depression_df = dataset_generation()
  df = load()
  # print(df[:10])
  tokenize(anxiety_df, cache_prefix = "anxiety_df")
  tokenize(control_df, cache_prefix = "control_df")
  tokenize(depression_df, cache_prefix = "depression_df")
