
from preprocessing import get_clean_datasets, dataset_generation
from lda_reddit_topics import train_reddit_topic_model
from roberta_embeddings import extract_roberta_embeddings
import os
if __name__ == "__main__":
    # Previous code remains the same until after stop words removal
    clean_control, clean_symptoms = get_clean_datasets()
    
    # Train LDA model
    lda_model, dictionary, corpus = train_reddit_topic_model(clean_control, clean_symptoms)
    
    # Save the trained model and dictionary (optional)
    cache_dir = 'cache_files'
    lda_model.save(os.path.join(cache_dir, 'lda_model'))
    dictionary.save(os.path.join(cache_dir, 'lda_dictionary'))

    control_embeddings = extract_roberta_embeddings(clean_control)
   
    symptom_embeddings = {
    for symptom, df in clean_symptoms.items():
        symptom_embeddings[symptom] = extract_roberta_embeddings(df)