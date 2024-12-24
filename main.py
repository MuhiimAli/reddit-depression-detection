
from preprocessing import get_clean_datasets, dataset_generation
from lda_reddit_topics import train_reddit_topic_model
from roberta_embeddings import extract_roberta_embeddings
import os
from evaluation import train_rf_models
import numpy as np
if __name__ == "__main__":
    # Previous code remains the same until after stop words removal
    clean_control, clean_symptoms = get_clean_datasets()
    
    # Train LDA model
    lda_model, dictionary, corpus = train_reddit_topic_model(clean_control, clean_symptoms)

    control_docs_idx = len(clean_control)

# Get LDA features for control
    num_topics = lda_model.num_topics  # Should be 200 based on your config

    control_lda_features = np.zeros((len(corpus[:control_docs_idx]), num_topics))
    for i, doc in enumerate(corpus[:control_docs_idx]):
        for topic_id, weight in lda_model.get_document_topics(doc):
            control_lda_features[i, topic_id] = weight

    symptom_lda_features = {}
    start_idx = control_docs_idx
    for symptom, df in clean_symptoms.items():
        end_idx = start_idx + len(df)
        feature_matrix = np.zeros((len(df), num_topics))
        for i, doc in enumerate(corpus[start_idx:end_idx]):
            for topic_id, weight in lda_model.get_document_topics(doc):
                feature_matrix[i, topic_id] = weight
        symptom_lda_features[symptom] = feature_matrix
        start_idx = end_idx
    
    # Save the trained model and dictionary (optional)
    cache_dir = 'cache_files'
    lda_model.save(os.path.join(cache_dir, 'lda_model'))
    dictionary.save(os.path.join(cache_dir, 'lda_dictionary'))

    # control_embeddings = extract_roberta_embeddings(clean_control)
   
    # symptom_embeddings = {}
    # for symptom, df in clean_symptoms.items():
    #     symptom_embeddings[symptom] = extract_roberta_embeddings(df)

    # results = train_rf_models(control_embeddings, symptom_embeddings)
    results = train_rf_models(control_lda_features, symptom_lda_features)

