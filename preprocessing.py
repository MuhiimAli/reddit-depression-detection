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
anxiety_reddits = ["Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack"]
anger_reddits = ["Anger"]
anhedonia_reddits = ["anhedonia", "DeadBedrooms"]
concentration_deficit_subreddits = ["DecisionMaking", "shouldi"]
disordered_eating_subreddits = ["bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous"]
fatigue_subreddits = ["chronicfatigue", "Fatigue"]
loneliness_subreddits = ["ForeverAlone", "lonely"]
sad_mood_subreddits = ["cry", "grief", "sad", "Sadness"]
self_loathing_subreddits = ["AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou"]
sleep_problem_subreddits = ["insomnia", "sleep"]
somatic_complaint_subreddits = ["cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus"]
suicidal_thoughts_subreddits = ["AdultSelfHarm", "selfharm", "SuicideWatch"]
worthlessness_subreddits = ["Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"]











def load():
  """Load pickles"""
  filepath = "/users/mali37/student.pkl"
  file = open(filepath,'rb')
  df = pd.read_pickle(file)
  return df
 



def dataset_generation(use_cache=True):
    """Build control and symptom datasets with improved performance"""
    # Create cache directory if it doesn't exist
    cache_dir = 'cache_files'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    symptom_datasets = {
        'anxiety': anxiety_reddits,
        'anger': anger_reddits,
        'anhedonia': anhedonia_reddits,
        'concentration_deficit': concentration_deficit_subreddits,
        'disordered_eating': disordered_eating_subreddits,
        'fatigue': fatigue_subreddits,
        'loneliness': loneliness_subreddits,
        'sad_mood': sad_mood_subreddits,
        'self_loathing': self_loathing_subreddits,
        'sleep_problems': sleep_problem_subreddits,
        'somatic_complaints': somatic_complaint_subreddits,
        'suicidal_thoughts': suicidal_thoughts_subreddits,
        'worthlessness': worthlessness_subreddits
    }
    
    # Update cache file paths
    all_cached = all(os.path.exists(os.path.join(cache_dir, f'{symptom}_dataset.pkl')) 
                    for symptom in symptom_datasets.keys())
    
    if use_cache and all_cached and os.path.exists(os.path.join(cache_dir, 'control_dataset.pkl')):
        print("Loading from cache...")
        symptom_dfs = {}
        for symptom in symptom_datasets.keys():
            cache_path = os.path.join(cache_dir, f'{symptom}_dataset.pkl')
            with open(cache_path, 'rb') as f:
                symptom_dfs[symptom] = pickle.load(f)
                # print(f"{symptom} dataset size: {len(symptom_dfs[symptom])}")
        
        control_path = os.path.join(cache_dir, 'control_dataset.pkl')
        with open(control_path, 'rb') as f:
            control_dataset = pickle.load(f)
            # print(f"Control dataset size: {len(control_dataset)}")
            # print(f"Number of unique users in control: {len(control_dataset['author'].unique())}")
        
        return symptom_dfs, control_dataset
    
    df = load()
    
    # Create symptom datasets
    symptom_dfs = {}
    for symptom, subreddits in symptom_datasets.items():
        symptom_dfs[symptom] = df[df['subreddit'].isin(subreddits)]
        # print(f"{symptom} dataset size: {len(symptom_dfs[symptom])}")
    
    depression_mask = df['subreddit'].isin(depression_subreddits)
    depression_posts = df[depression_mask]
    
    user_first_depression_post = (
        depression_posts
        .groupby('author')['created_utc']
        .min()
    )
    
    df['first_depression_post'] = df['author'].map(user_first_depression_post)
    cutoff_dates = df['first_depression_post'] - (180 * 24 * 60 * 60)
    
    control_mask = (
        (~depression_mask) &
        (df['created_utc'] < cutoff_dates) &
        (~df['first_depression_post'].isna())
    )
    
    control_dataset = df[control_mask]
    
    if use_cache:
        print("Saving to cache...")
        for symptom, dataset in symptom_dfs.items():
            cache_path = os.path.join(cache_dir, f'{symptom}_dataset.pkl')
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
        
        control_path = os.path.join(cache_dir, 'control_dataset.pkl')
        with open(control_path, 'wb') as f:
            pickle.dump(control_dataset, f)
    
    # print(f"Control dataset size: {len(control_dataset)}")
    # print(f"Number of unique users in control: {len(control_dataset['author'].unique())}")
    
    return symptom_dfs, control_dataset
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
    cache_dir = 'cache_files'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f'{cache_prefix}_tokenized.pkl')
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading tokenized {cache_prefix} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # print(f"Tokenizing {cache_prefix} dataset...")
    
    tokenized_df = dataset.copy()
    tokenized_df['tokenized_text'] = tokenized_df['text'].apply(tokenize_text)
    tokenized_df['token_count'] = tokenized_df['tokenized_text'].apply(len)
    tokenized_df = tokenized_df[tokenized_df['token_count'] > 0]
    
    if use_cache:
        print(f"Saving tokenized {cache_prefix} to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_df, f)
    
    return tokenized_df


def stop_words(control_df):
    """Find top 100 words from control dataset to use as stop words"""
    print("Generating stop words from control dataset...")
    
    # Flatten all tokens from control dataset into a single list
    all_tokens = []
    for tokens in control_df['tokenized_text']:
        all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = pd.Series(all_tokens).value_counts()
    
    # Get top 100 most frequent tokens
    top_100_words = set(token_counts.head(100).index.tolist())
    # print(f"Top 100 stop words: {top_100_words}")
    
    return top_100_words
def remove_stop_words(df, stop_words_set):
    """Remove stop words from a dataframe's tokenized_text"""
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Remove stop words from tokenized_text
    clean_df['tokenized_text'] = clean_df['tokenized_text'].apply(
        lambda tokens: [token for token in tokens if token not in stop_words_set]
    )
    
    # Update token count
    clean_df['token_count'] = clean_df['tokenized_text'].apply(len)
    
    # Remove empty posts after stop word removal
    clean_df = clean_df[clean_df['token_count'] > 0]
    
    return clean_df

def get_clean_datasets():
    symptom_datasets, control_df = dataset_generation()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_control = tokenize(control_df, cache_prefix="control_df")
    tokenized_symptoms = {}
    for symptom, df in symptom_datasets.items():
        tokenized_symptoms[symptom] = tokenize(df, cache_prefix=f"{symptom}_df")
    
    # Generate stop words from control dataset and remove from all datasets
    stop_words_set = stop_words(tokenized_control)
    
    clean_control = remove_stop_words(tokenized_control, stop_words_set)
    # print(f"Control dataset size after stop word removal: {len(clean_control)}")
    
    clean_symptoms = {}
    for symptom, df in tokenized_symptoms.items():
        clean_symptoms[symptom] = remove_stop_words(df, stop_words_set)
        # print(f"{symptom} dataset size after stop word removal: {len(clean_symptoms[symptom])}")
    return clean_control, clean_symptoms
    
