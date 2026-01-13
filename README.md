# NLP Assignment 1: That’s What I LIKE

## Overview
This repository contains my implementation for NLP Assignment 1. I have implemented three types of Word Embedding models (Skipgram, Negative Sampling, and GloVe) from scratch using PyTorch and compared them with a pre-trained Gensim model. Additionally, I developed a Context Search Engine web application.

## Features
- **Word Embedding Models**: 
  - Word2Vec (Skipgram) - implemented from scratch.
  - Word2Vec (Negative Sampling) - implemented from scratch.
  - GloVe (Global Vectors) - implemented from scratch.
- **Evaluation**: 
  - Syntactic and Semantic Accuracy using Word Analogies.
  - Correlation analysis using WordSim-353 dataset.
  - Comparison with pre-trained GloVe (Twitter-25) from Gensim.
- **Web Application**: A Flask-based search engine that finds relevant context based on semantic similarity.

## Directory Structure
- `NLP_Assignment_Complete.ipynb`: **Main Submission File**. Comprehensive notebook (CPU/Standard) containing all code.
- `NLP_Assignment_Colab.ipynb`: **GPU-Optimized Notebook**. Use this for Google Colab or Mac M1/M2 acceleration.
- `app/`: Contains the Flask web application code and templates.
- `models/`: Stores the trained PyTorch models and training metrics.
- `wordsim353/`: Dataset for correlation evaluation.
- `word-test.v1.txt`: Dataset for analogy evaluation.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Notebook**:
   - Open `NLP_Assignment_Complete.ipynb` (or the Colab version).
   - Run all cells to see the training process, evaluation tables, and to launch the web application demo.
   - **Note**: The notebook will automatically download the required NLTK Reuters corpus to a local `nltk_data` folder on the first run.

3. **Run Web App Standalone**:
   - You can also run the web app directly from the terminal:
     ```bash
     python app/app.py
     ```
