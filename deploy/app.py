
from flask import Flask, render_template, request
import numpy as np
import json
import os

app = Flask(__name__)

# Load Artifacts
print("Loading Artifacts...")
base_dir = os.path.dirname(__file__)

try:
    embeddings = np.load(os.path.join(base_dir, 'embeddings.npy'))
    corpus_embeddings = np.load(os.path.join(base_dir, 'corpus_embeddings.npy'))
    
    with open(os.path.join(base_dir, 'word2index.json'), 'r') as f:
        word2index = json.load(f)
        
    with open(os.path.join(base_dir, 'corpus.json'), 'r') as f:
        corpus = json.load(f)
        
    print("Artifacts loaded.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    embeddings = None
    corpus_embeddings = None
    word2index = {}
    corpus = []

EMB_SIZE = embeddings.shape[1] if embeddings is not None else 10

def get_query_embedding(query):
    if embeddings is None: return np.zeros(EMB_SIZE)
    tokens = query.lower().split()
    indices = [word2index.get(w, word2index.get('<UNK>', 0)) for w in tokens]
    if not indices:
        return np.zeros(EMB_SIZE)
    vecs = embeddings[indices]
    return np.mean(vecs, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query and corpus_embeddings is not None:
            # Get query vector
            q_vec = get_query_embedding(query)
            
            # Normalize
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
                
            # Search
            scores = np.dot(corpus_embeddings, q_vec)
            top_indices = np.argsort(scores)[-10:][::-1]
            
            for idx in top_indices:
                results.append({
                    'score': float(scores[idx]),
                    'text': corpus[idx]
                })
                
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
