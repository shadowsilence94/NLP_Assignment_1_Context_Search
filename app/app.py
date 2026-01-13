
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import nltk
from nltk.corpus import reuters
from collections import Counter
import zipfile

app = Flask(__name__)

# ==========================================
# 1. Inline Data Loader Logic (Self-Contained)
# ==========================================

# Constants
MIN_FREQ = 5 

class DataLoader:
    def __init__(self, min_freq=MIN_FREQ):
        self.min_freq = min_freq
        self.categories = None
        print(f"Loading Reuters corpus (Full)...")
        
        # Ensure NLTK data path
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)

        # Download/Unzip if needed
        try:
            nltk.data.find('corpora/reuters')
        except LookupError:
            print("Downloading reuters corpus...")
            nltk.download('reuters', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        
        reuters_zip_path = os.path.join(nltk_data_dir, 'corpora', 'reuters.zip')
        reuters_dir_path = os.path.join(nltk_data_dir, 'corpora', 'reuters')
        if os.path.exists(reuters_zip_path) and not os.path.exists(reuters_dir_path):
             print(f"Unzipping {reuters_zip_path}...")
             with zipfile.ZipFile(reuters_zip_path, 'r') as zip_ref:
                 zip_ref.extractall(os.path.join(nltk_data_dir, 'corpora'))
             print("Unzipping complete.")
        
        # Get sentences
        self.sentences = reuters.sents()
        self.corpus = [[word.lower() for word in sent] for sent in self.sentences]
        
        print(f"Corpus size: {len(self.corpus)} sentences")
        self.build_vocab()
        
    def build_vocab(self):
        print("Building vocabulary...")
        flatten = lambda l: [item for sublist in l for item in sublist]
        self.all_words = flatten(self.corpus)
        self.word_count = Counter(self.all_words)
        
        self.vocab = [w for w, c in self.word_count.items() if c >= self.min_freq]
        self.vocab.append('<UNK>')
        
        self.word2index = {w: i for i, w in enumerate(self.vocab)}
        self.index2word = {i: w for w, i in self.word2index.items()}
        self.voc_size = len(self.vocab)
        print(f"Vocabulary size: {self.voc_size}")
        
    def get_word2index(self):
        return self.word2index

# ==========================================
# 2. Inline Model Definition (Self-Contained)
# ==========================================

class SkipgramNeg(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid        = nn.LogSigmoid()
    
    def forward(self, center, outside, negative):
        # Inference only needs embeddings, forward not used here
        pass

# ==========================================
# 3. App Logic
# ==========================================

# Load Data and Models
print("Loading Data...")
loader = DataLoader()
word2index = loader.get_word2index()
index2word = {v:k for k,v in word2index.items()}
corpus = loader.corpus # lowercased list of lists
vocab = loader.vocab

device = torch.device('cpu') # Deployment usually CPU
EMB_SIZE = 10
VOC_SIZE = loader.voc_size

print("Loading Models...")
model = SkipgramNeg(VOC_SIZE, EMB_SIZE)
model_path = os.path.join(os.path.dirname(__file__), '../models/skipgram_neg_model.pth')

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Model loaded.")
    else:
        print(f"Warning: Model not found at {model_path}")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Cache embeddings
embeddings = None
if model:
    # v_c + v_o / 2
    embeddings = (model.embedding_center.weight + model.embedding_outside.weight).detach().numpy() / 2
    # Normalize
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norm + 1e-9)

def get_sentence_embedding(sentence):
    indices = [word2index.get(w, word2index['<UNK>']) for w in sentence]
    if not indices:
        return np.zeros(EMB_SIZE)
    vecs = embeddings[indices]
    return np.mean(vecs, axis=0)

# Pre-compute sentence embeddings
print("Pre-computing corpus embeddings...")
corpus_embeddings = []
if embeddings is not None:
    # Use a subset if corpus is too large? 50k is borderline for instant startup but fine.
    for sent in corpus:
        corpus_embeddings.append(get_sentence_embedding(sent))
    corpus_embeddings = np.array(corpus_embeddings) 
    norm_corpus = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_embeddings = corpus_embeddings / (norm_corpus + 1e-9)
    print("Pre-computation ready.")
else:
    print("Model missing. Search disabled.")
    corpus_embeddings = None


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query and corpus_embeddings is not None:
            q_tokens = query.lower().split()
            q_vec = get_sentence_embedding(q_tokens)
            
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
                
            scores = np.dot(corpus_embeddings, q_vec)
            top_indices = np.argsort(scores)[-10:][::-1]
            
            for idx in top_indices:
                results.append({
                    'score': float(scores[idx]),
                    'text': " ".join(corpus[idx])
                })
                
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
