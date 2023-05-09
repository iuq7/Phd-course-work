import math
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# Parse the term index file
term_index = {}
with open(os.path.join(script_dir, 'indices', 'term_index.txt'), 'r') as f:
    for line in f:
        index, term = line.strip().split()
        term_index[term] = int(index)

# Parse the document index file
doc_index = {}
with open(os.path.join(script_dir, 'indices', 'document_index.txt'), 'r') as f:
    for line in f:
        index, filename = line.strip().split()
        doc_index[int(index)] = filename

# Parse the inverted index file
inverted_index = {}
with open(os.path.join(script_dir, 'indices', 'inverted_index.txt'), 'r') as f:
    for line in f:
        temp = line.strip().split()
        term, postings_list = temp[0], temp[1:]
        postings = {}
        for posting in postings_list:
            doc_id, tfidf = posting.split(':')
            postings[int(doc_id)] = float(tfidf)
        inverted_index[int(term)] = postings

def doSearch(query):
    # Tokenize the query
    terms = query.strip().split()
    terms = [lemmatizer.lemmatize(stemmer.stem(word)) for word in terms if word.isalpha() and len(word) > 2 and word not in stop_words]
    # Compute the query vector
    query_vector = {}
    for term in terms:
        if term not in term_index:
            continue
        term_id = term_index[term]
        query_vector[term_id] = query_vector.get(term_id, 0) + 1

    # Normalize the query vector
    query_norm = math.sqrt(sum(x*x for x in query_vector.values()))
    for term_id in query_vector:
        query_vector[term_id] /= query_norm
        
    # Compute the cosine similarity between the query vector and each document
    results = []
    for doc_id in doc_index:
        doc_vector = {}
        for term_id in query_vector:
            if term_id in inverted_index and doc_id in inverted_index[term_id]:
                doc_vector[term_id] = inverted_index[term_id][doc_id]
        # Compute the cosine similarity
        dot_product = sum(query_vector[term_id] * doc_vector[term_id] for term_id in doc_vector)
        doc_norm = math.sqrt(sum(x*x for x in doc_vector.values()))
        if doc_norm == 0:
            continue
        similarity = dot_product / doc_norm

        # Add the result to the list
        results.append((doc_id, similarity))
    # Sort the results by decreasing cosine similarity
    results.sort(key=lambda x: x[1], reverse=True)

    # Write the results to a new file
    with open(os.path.join(script_dir, 'indices', 'bm25_cosine_weights.txt'), 'w') as f:
        f.write(f"Query: {query}\n")
        f.write("Wheighting Scheme: BM25\n")
        f.write("indexNo. Of Document  document-filename  cosine similarity weight\n")
        for doc_id, similarity in results:
            filename = doc_index[doc_id]
            f.write(f"{doc_id} {filename} {similarity}\n")
    return 0