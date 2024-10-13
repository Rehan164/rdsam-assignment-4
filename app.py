from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here

# Preload the dataset, vectorizer, and LSA components once
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(newsgroups.data)

# Use TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)
X_lsa = svd.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    """
    Function to search for top 5 similar documents given a query.
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_vectorized = vectorizer.transform([query])
    query_lsa_transformed = svd.transform(query_vectorized)
    similarity_scores = cosine_similarity(query_lsa_transformed, X_lsa).flatten()
    best_match_indices = np.argsort(similarity_scores)[-5:][::-1]
    matching_documents = [newsgroups.data[i] for i in best_match_indices]
    top_similarities = [similarity_scores[i] for i in best_match_indices]
    result_indices = best_match_indices.tolist()

    return matching_documents, top_similarities, result_indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)

