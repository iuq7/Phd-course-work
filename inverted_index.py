import os
import math
from collections import Counter
from pdfminer.high_level import extract_text
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

# Initialize stemmer and lemmatizer
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# Set up stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))

# Set up BM25 parameters
k1 = 1.2
b = 0.75
avg_doc_len = 500  # average length of documents in the collection

# Set up file paths
data_dir = "/Users/irfan/Documents/PhD/sem1/AIR/AIRT_Assignment_1_Irfan_Task#1/corpus/"
output_dir = "/Users/irfan/Documents/PhD/sem1/AIR/AIRT_Assignment_1_Irfan_Task#3/"
inverted_index_file = os.path.join(output_dir, "inverted_index.txt")
term_index_file = os.path.join(output_dir, "term_index.txt")
doc_index_file = os.path.join(output_dir, "document_index.txt")

# Initialize data structures
term_index = {}
doc_index = {}
inverted_index = {}

# Loop over all PDF files in the data directory
for f_index, filename in enumerate(os.listdir(data_dir)):
    if filename.endswith(".pdf"):
        # Open the PDF file
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            # Parse the PDF file
            text = extract_text(filepath)
            # Tokenize the text
            tokens = nltk.word_tokenize(text.lower())
            # Remove punctuation and short words, and apply stemming and lemmatization
            tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]
            # Count the term frequencies
            term_freqs = Counter(tokens)
            # Update the inverted index and document index
            doc_id = len(doc_index) + 1
            doc_index[doc_id] = filename
            doc_len = sum(term_freqs.values())
            for term, freq in term_freqs.items():
                if term not in term_index:
                    term_index[term] = len(term_index)
                term_id = term_index[term]
                if term_id not in inverted_index:
                    inverted_index[term_id] = {}
                tf = freq/doc_len
                idf = math.log((len(doc_index)+1)/(len(inverted_index[term_id])+1))
                inverted_index[term_id][doc_id] = tf*(k1+1)/(tf+k1*(1-b+b*doc_len/avg_doc_len))*idf

# Write the output files
with open(inverted_index_file, "w") as f:
    for term_id, postings in inverted_index.items():
        posting_list = " ".join([f"{doc_id}:{score:.3f}" for doc_id, score in postings.items()])
        f.write(f"{term_id} {posting_list}\n")

with open(term_index_file, "w") as f:
    for term, term_id in term_index.items():
        f.write(f"{term_id} {term}\n")

with open(doc_index_file, "w") as f:
    for doc_id, filename in doc_index.items():
        f.write(f"{doc_id} {filename}\n")

print("ALL DONE")
