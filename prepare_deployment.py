
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys

import nltk
from nltk.corpus import reuters
from collections import Counter
import zipfile

# 1. Setup Data
print("Loading NLTK Data...")
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Force downloads
for res in ['reuters', 'punkt', 'punkt_tab']:
    try:
        if res == 'reuters':
            nltk.data.find('corpora/reuters')
        elif res == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif res == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print(f"Downloading {res}...")
        nltk.download(res, download_dir=nltk_data_dir, quiet=True)

# Unzip reuters if needed (explicit check)
reuters_zip_path = os.path.join(nltk_data_dir, 'corpora', 'reuters.zip')
reuters_dir_path = os.path.join(nltk_data_dir, 'corpora', 'reuters')
if os.path.exists(reuters_zip_path) and not os.path.exists(reuters_dir_path):
    print("Unzipping reuters...")
    with zipfile.ZipFile(reuters_zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(nltk_data_dir, 'corpora'))

corpus = [[word.lower() for word in sent] for sent in reuters.sents()]
print(f"Corpus Size: {len(corpus)}")

# Vocab
print("Building Vocab...")
counter = Counter([w for sent in corpus for w in sent])
vocab = [w for w, c in counter.items() if c >= 5] + ['<UNK>']
word2index = {w:i for i, w in enumerate(vocab)}
print(f"Vocab Size: {len(vocab)}")

# 2. Setup Model
class SkipgramNeg(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)

EMB_SIZE = 10
model = SkipgramNeg(len(vocab), EMB_SIZE)

model_path = 'models/skipgram_neg_model.pth'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please train the model first!")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
print("Model loaded.")

# 3. Export Embeddings
# v_c + v_o / 2
print("Exporting Word Embeddings...")
embeddings = (model.embedding_center.weight + model.embedding_outside.weight).detach().numpy() / 2
# Normalize
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norm + 1e-9)

# 4. Export Corpus Embeddings
print("Exporting Corpus Embeddings...")
unk_idx = word2index['<UNK>']
corpus_embeddings = []

for sent in corpus:
    indices = [word2index.get(w, unk_idx) for w in sent]
    if not indices:
        vec = np.zeros(EMB_SIZE)
    else:
        vec = np.mean(embeddings[indices], axis=0)
    corpus_embeddings.append(vec)

corpus_embeddings = np.array(corpus_embeddings)
# Normalize corpus
cnorm = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
corpus_embeddings = corpus_embeddings / (cnorm + 1e-9)

# 5. Save to deploy folder
os.makedirs('deploy', exist_ok=True)
os.makedirs('deploy/static', exist_ok=True)
os.makedirs('deploy/templates', exist_ok=True)

# Save NPY
np.save('deploy/embeddings.npy', embeddings)
np.save('deploy/corpus_embeddings.npy', corpus_embeddings)

# Save JSONs
with open('deploy/word2index.json', 'w') as f:
    json.dump(word2index, f)

# Save Corpus (Text) - List of strings (joined sentences) to save space compared to list of lists
print("Saving Corpus Text...")
corpus_text = [" ".join(sent) for sent in corpus]
with open('deploy/corpus.json', 'w') as f:
    json.dump(corpus_text, f)

print("Done! Artifacts saved to deploy/")
