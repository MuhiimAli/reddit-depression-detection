from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
import logging
import os
import numpy as np
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
    # print("\nTop 10 words in each topic:")
    # for idx, topic in lda_model.print_topics(-1):
    #     print(f'Topic {idx}: {topic}')
    
    return lda_model, dictionary, corpus

def get_lda_results(lda_model, dictionary, corpus, clean_control, clean_symptoms):
    control_docs_idx = len(clean_control)
    num_topics = lda_model.num_topics

    # Control features
    control_features = np.zeros((len(corpus[:control_docs_idx]), num_topics))
    for i, doc in enumerate(corpus[:control_docs_idx]):
        for topic_id, weight in lda_model.get_document_topics(doc):
            control_features[i, topic_id] = weight

    # Symptom features
    symptom_features = {}
    start_idx = control_docs_idx
    for symptom, df in clean_symptoms.items():
        end_idx = start_idx + len(df)
        features = np.zeros((len(df), num_topics))
        for i, doc in enumerate(corpus[start_idx:end_idx]):
            for topic_id, weight in lda_model.get_document_topics(doc):
                features[i, topic_id] = weight
        symptom_features[symptom] = features
        start_idx = end_idx

    return {"control": control_features, "symptoms": symptom_features}