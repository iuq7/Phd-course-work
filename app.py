from flask import Flask, request, render_template, send_file
import os
from search_engine import doSearch
# import search_engine

script_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    result = doSearch(query)
    return send_file(
        os.path.join(script_dir, 'indices', 'bm25_cosine_weights.txt'),
        mimetype='text/plain',
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)
