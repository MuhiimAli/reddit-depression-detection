from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
def train_rf_models(control_embeddings, symptom_embeddings):
    """Train binary Random Forest classifier for each symptom"""
    results = {}
    total = len(symptom_embeddings)
   
    for i, (symptom, symptom_emb) in enumerate(symptom_embeddings.items(), 1):
        print(f"\rTraining model for {symptom} ({i}/{total})", end="")
        
        # Prepare data
        X = np.vstack([control_embeddings, symptom_emb])
        y = np.array([0] * len(control_embeddings) + [1] * len(symptom_emb))
        
        rf_classifier = RandomForestClassifier(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = cross_validate(rf_classifier, X=X, y=y, cv=cv, 
                                    scoring='roc_auc', return_train_score=True)
        
        results[symptom] = {
            'train_scores': cv_results['train_score'],
            'test_scores': cv_results['test_score']
        }
        
        print(f"\n{symptom}:")
        print(f"Train ROC-AUC: {cv_results['train_score'].mean():.3f} ± {cv_results['train_score'].std():.3f}")
        print(f"Test ROC-AUC: {cv_results['test_score'].mean():.3f} ± {cv_results['test_score'].std():.3f}")
   
    print("\nCompleted training all models")
    return results
