
from preprocessing import get_clean_datasets, dataset_generation
from lda_reddit_topics import train_reddit_topic_model, get_lda_results
from roberta_embeddings import extract_roberta_embeddings
import os
from evaluation import train_rf_models
import numpy as np

def compare_results(my_results):
    """Compare my results with TA solution results"""
    ta_results = {
       'anger': {'lda': 0.794, 'roberta': 0.928},
       'anhedonia': {'lda': 0.906, 'roberta': 0.956}, 
       'anxiety': {'lda': 0.837, 'roberta': 0.952},
       'disordered_eating': {'lda': 0.905, 'roberta': 0.952},
       'loneliness': {'lda': 0.806, 'roberta': 0.907},
       'sad_mood': {'lda': 0.788, 'roberta': 0.919},
       'self_loathing': {'lda': 0.815, 'roberta': 0.922},
       'sleep_problems': {'lda': 0.909, 'roberta': 0.956},
       'somatic_complaints': {'lda': 0.880, 'roberta': 0.925},
       'worthlessness': {'lda': 0.700, 'roberta': 0.897}
   }
    for symptom in ta_results:
        if symptom in my_results:
            ta_lda = ta_results[symptom]['lda'] 
            ta_roberta = ta_results[symptom]['roberta']
            my_lda = my_results[symptom]['lda']
            my_roberta = my_results[symptom]['roberta']
            
            print(f"\n{symptom}:")
            print(f"LDA: {my_lda:.3f} (TA: {ta_lda:.3f}) {'PASS' if ta_lda-0.2 <= my_lda <= ta_lda+0.2 else 'FAIL'}")
            print(f"RoBERTa: {my_roberta:.3f} (TA: {ta_roberta:.3f}) {'PASS' if ta_roberta-0.2 <= my_roberta <= ta_roberta+0.2 else 'FAIL'}")

if __name__ == "__main__":
   # Get datasets
    clean_control, clean_symptoms = get_clean_datasets()
#   # Get LDA features
    lda_results = get_lda_results(clean_control, clean_symptoms)
   
    control, symptoms = dataset_generation()
     # Get RoBERTa features 
    control_embeddings = extract_roberta_embeddings(control, cache_prefix='control')
    symptom_embeddings = {}
    for symptom, df in symptoms.items():
        symptom_embeddings[symptom] = extract_roberta_embeddings(df, cache_prefix=symptom)
   
   # Train and evaluate models
    lda_scores = train_rf_models(lda_results["control"], lda_results["symptoms"])
    roberta_scores = train_rf_models(control_embeddings, symptom_embeddings)
   
   # Compare with TA solution results
    results = {
       symptom: {
           "lda": lda_scores[symptom]["test_scores"].mean(),
           "roberta": roberta_scores[symptom]["test_scores"].mean()
       }
       for symptom in clean_symptoms.keys()
   }
    compare_results(results)
