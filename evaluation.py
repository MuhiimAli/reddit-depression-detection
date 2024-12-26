from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def process_single_symptom(symptom, symptom_emb, control_embeddings):
    """Process a single symptom in parallel"""
    X = np.vstack([control_embeddings, symptom_emb])
    y = np.array([0] * len(control_embeddings) + [1] * len(symptom_emb))
    
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # Use 1 here since we're parallelizing at a higher level
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(rf_classifier, X=X, y=y, cv=cv, 
                              scoring='roc_auc', return_train_score=True)
    
    return symptom, cv_results['train_score'], cv_results['test_score']

def train_rf_models(control_embeddings, symptom_embeddings):
    """Train models in parallel"""
    # Run all symptoms in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_symptom)(
            symptom, symptom_emb, control_embeddings
        )
        for symptom, symptom_emb in symptom_embeddings.items()
    )
    
    # Format results
    formatted_results = {}
    for symptom, train_scores, test_scores in results:
        print(f"\n{symptom}:")
        print(f"Train ROC-AUC: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
        print(f"Test ROC-AUC: {test_scores.mean():.3f} ± {test_scores.std():.3f}")
        
        formatted_results[symptom] = {
            'train_scores': train_scores,
            'test_scores': test_scores
        }
    
    print("\nCompleted training all models")
    return formatted_results
