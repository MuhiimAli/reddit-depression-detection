from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def extract_roberta_embeddings(df, model_name='distilroberta-base', layer_idx=5):
   """Extract embeddings from 5th layer of DistilRoBERTa for each post"""
   
   # Initialize tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
   
   # Move model to GPU if available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   model.eval()
   
   embeddings = []
   
   # Process each text
   for text in df['text']: #TODO are we suppose to use the tokenized text here?
       # Tokenize
       inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
       inputs = {k: v.to(device) for k, v in inputs.items()}
       
       # Get embeddings from specified layer
       with torch.no_grad():
           outputs = model(**inputs)
           # Get hidden states from layer_idx (5th layer)
           hidden_states = outputs.hidden_states[layer_idx]

           #TODO: Oh, yeah, run them in batches and average over the sequence dimension, 
           # so you end up with something like [32, 768]. Do this for all posts so that at the end you have [total posts, 768]
        #    print(f"hidden_states: {hidden_states.shape}")
           
           # Average token embeddings to get post embedding
           post_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        #    print(f"Shape: {post_embedding.shape}")
           embeddings.append(post_embedding)
   
   return np.array(embeddings)