import streamlit as st
import pandas as pd
import numpy as np

from inference.acrobert import acronym_linker
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import yake
import os
import sys

# Setting path for Huggingface Models to get downloaded
os.environ['TRANSFORMERS_CACHE'] = 'path\to\Huggingface_cache'

# Loads pre-trained models for sentence embedding and zero-shot classification.
@st.cache_resource
def load_models():

    LABELS = ['skills', 'college_name', 'degree', 'experience', 'company_names']
    MODEL_NAME = "all-MiniLM-L6-v2"
    sentence_model = SentenceTransformer(MODEL_NAME)
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return sentence_model, zero_shot_classifier, LABELS

# Loads and processes data from a CSV file containing resume details (extracting from pdf files).
@st.cache_resource
def load_data(_sentence_model):
    #Processing Data
    csv_data = pd.read_csv("path/to/resume.csv file") # you can raise a request to get the file from here https://drive.google.com/file/d/1I8x7rL7T4W1A5TPJLwqUK9LWaadtI7vq/view?usp=drive_link
    resumes = csv_data['resume_details']
    resumes = [eval(resume) for resume in resumes]
    
    exp1 = [(cv['experience'], cv['name']) if cv['experience'] is not None else ('Not Found', cv['name']) for cv in resumes]
    exp2 = [" ".join(text_samp for text_samp in res_text[0]) for res_text in exp1]
    sent_exp = [sent_tokenize(exp_text) for exp_text in exp2]

    exp_embed2 = []

    #Generating Embeddings
    for idx, exp in enumerate(tqdm(sent_exp, desc="Encoding experiences")):
        exp_embed = [sentence_model.encode(exp_text) for exp_text in exp]
        exp_embed2.append(exp_embed)
    
    return resumes, exp_embed2

# Classifies a User's query into predefined categories using zero-shot classification.
def classify_query(query, zero_shot_classifier, LABELS):
    result = zero_shot_classifier(query, LABELS)
    return result['labels'][:3]

# Extracts keywords from a given query using the YAKE keyword extraction algorithm.
def extract_keywords(query, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(lan='en', n=1)
    keywords = kw_extractor.extract_keywords(query)
    return [keyword for keyword, _ in keywords][:max_keywords]

# Using BM25 search to retrieve relevant documents based on a given query.
def bm25_search(query, documents, fields, top_k=3):
    documents_new = []
    for cv in documents:
        combined_text = []
        for field in fields:
            if cv[field] is not None:
                combined_text.extend(cv[field])
        if not combined_text:
            documents_new.append("")
        else:
            documents_new.append(" ".join(text_samp for text_samp in combined_text).lower())
    
    tokenized_corpus = [doc.split() for doc in documents_new]
    
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(doc_scores)[-top_k:][::-1]
    return top_k_indices, doc_scores[top_k_indices]

# Calculates the cosine similarity between two embeddings.
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Performing semantic search to find the most semantically similar embeddings to a given query embedding.
def semantic_search(query_embedding, nested_list_embeddings, n):
    scores = []
    for idx, embeddings_list in enumerate(tqdm(nested_list_embeddings, desc="Semantic Search Progress")):
        for emb in embeddings_list:
            score = calculate_similarity(query_embedding, emb)
            scores.append((score, idx, np.where(embeddings_list == emb)[0][0]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [x[1:] for x in scores[:n]]

# Expands the User query by replacing acronyms with their full forms.
def expand_query(query):
    expanded_terms = acronym_linker(query, mode='acrobert')
    for acronym, full_form in expanded_terms:
        query = query.replace(acronym, full_form + "("+acronym+")")
    return query

# Streamlit app
st.title("Resume Search App")

# User input for query and number of results
query = st.text_input("Enter your query:")
n_results = st.slider("Number of results:", 1, 10, 5)

# Load models and data if not already loaded in session state
if 'sentence_model' not in st.session_state:
    sentence_model, zero_shot_classifier, LABELS = load_models()
    resumes, exp_embed2 = load_data(sentence_model)
    st.session_state.sentence_model = sentence_model
    st.session_state.zero_shot_classifier = zero_shot_classifier
    st.session_state.LABELS = LABELS
    st.session_state.resumes = resumes
    st.session_state.exp_embed2 = exp_embed2

# Perform search when the search button is clicked
if st.button("Search"):
    if query:
        st.write("Searching for resumes...")

        # Expand and classify the query
        expanded_query = expand_query(query)
        classified_labels = classify_query(expanded_query.lower(), st.session_state.zero_shot_classifier, st.session_state.LABELS)

        # Divide the number of results between BM25 and semantic search
        bm25_n = n_results // 2
        semantic_n = n_results - bm25_n

        # Perform BM25 search
        bm25_indices, _ = bm25_search(expanded_query.lower(), st.session_state.resumes, classified_labels, bm25_n)

        # Check if BM25 search failed
        if len(bm25_indices) == 0:
            st.error("BM25 search failed due to empty corpus. Please check the input data.")
        else:
            # Encode query and extract keywords for semantic search
            query_embedding = st.session_state.sentence_model.encode(expanded_query)
            keywords = extract_keywords(expanded_query)
            keyword_embeddings = [st.session_state.sentence_model.encode(word) for word in keywords]
            #combining query & keyword embeddings
            combined_query_embedding = np.mean([query_embedding] + keyword_embeddings, axis=0)

            # Perform semantic search
            semantic_indices = semantic_search(combined_query_embedding, st.session_state.exp_embed2, semantic_n)
            # Combine indices from semantic and BM25 search
            combined_indices = semantic_indices + list(bm25_indices) 

            # Display top resume matches
            st.write("Top Resume Matches:")
            for idx in combined_indices:
                if isinstance(idx,tuple):
                    st.write(st.session_state.resumes[idx[0]])
                else:
                    st.write(st.session_state.resumes[idx])
    else:
        st.write("Please enter a query to search.")
