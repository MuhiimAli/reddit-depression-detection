from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
import logging
import os

def train_reddit_topic_model(clean_control, clean_symptoms, num_topics=10):
    """Train LDA model on the entire datasets"""
    print("Training LDA model...")
    
    # Combine all documents for training
    all_docs = []
    
    # Add control documents
    all_docs.extend(clean_control['tokenized_text'].tolist())
    
    # Add symptom documents
    for symptom, df in clean_symptoms.items():
        all_docs.extend(df['tokenized_text'].tolist())
    
    # Create dictionary
    dictionary = Dictionary(all_docs)
    
    
    # Convert documents to bag of words
    corpus = [dictionary.doc2bow(doc) for doc in all_docs]
    
    # Set up logging for model training
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # Train LDA model using multicore implementation
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        workers=3, 
        passes=10,
        random_state=42  # For reproducibility
    )
    
    # Print topics
    print("\nTop 10 words in each topic:")
    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic {idx}: {topic}')
    
    return lda_model, dictionary, corpus