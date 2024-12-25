from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import pickle

def extract_roberta_embeddings(df, cache_prefix='', use_cache=True, model_name='distilroberta-base', layer_idx=5, batch_size=32):
   """Extract embeddings from 5th layer of DistilRoBERTa for each post"""
   cache_dir = 'cache_files'
   cache_file = os.path.join(cache_dir, f'{cache_prefix}_roberta_embeddings.pkl')
   
   if use_cache and os.path.exists(cache_file):
       print(f"Loading cached RoBERTa embeddings for {cache_prefix}...")
       with open(cache_file, 'rb') as f:
           embeddings = pickle.load(f)
           return embeddings

   print(f"Extracting RoBERTa embeddings for {cache_prefix}...")
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   model.eval()
   
   all_embeddings = []
   total = len(df)
   
   for i in range(0, total, batch_size):
       batch_texts = df['text'].iloc[i:i+batch_size].tolist()
       print(f"\rProcessing batch {i//batch_size + 1}/{(total-1)//batch_size + 1}", end="")
       
       inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                        max_length=512, return_tensors='pt')
       inputs = {k: v.to(device) for k, v in inputs.items()}
       
       with torch.no_grad():
           outputs = model(**inputs)
           hidden_states = outputs.hidden_states[layer_idx]
           batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
           all_embeddings.extend(batch_embeddings)

   embeddings = np.array(all_embeddings)
   print(f"\nExtracted embeddings shape: {embeddings.shape}")
   
   if use_cache:
       os.makedirs(cache_dir, exist_ok=True)
       print(f"Caching embeddings for {cache_prefix}...")
       with open(cache_file, 'wb') as f:
           pickle.dump(embeddings, f)
   
   return embeddings