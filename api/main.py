import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import preprocess 
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#create an object of the class Flask
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
word2vec = pickle.load(open('word2vec.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])


def predict():
    text = (request.form['judul'])
    # Pra-pemrosesan teks
    processed_text = preprocess.preprocess(text)
    # Lakukan transformasi TF-IDF
    tfidf_features = tfidf.transform([processed_text])
    # Lakukan transformasi Word2Vec
    word2vec_features = transform_word2vec(processed_text, word2vec)
    # Gabungkan fitur-fitur dari TF-IDF dan Word2Vec
    combined_features = combine_features(tfidf_features, word2vec_features)
    # Lakukan prediksi
    prediction = model.predict(combined_features.reshape(1, -1))[0]
    # Hasil 
    return render_template('index.html',prediction_text = f'buku berjudul "{text}" termasuk klasifikasi nomor {prediction}')


def transform_word2vec(text, word2vec):
    # Lakukan tokenisasi
    tokens = word_tokenize(text)
    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian', 'english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Ambil vektor rata-rata dari vektor kata Word2Vec
    word2vec_features = np.mean([word2vec.wv[word] for word in tokens if word in word2vec.wv], axis=0)
    return word2vec_features


def combine_features(tfidf_features, word2vec_features):
    # Gabungkan fitur-fitur dari TF-IDF dan Word2Vec
    combined_features = np.concatenate([tfidf_features.toarray(), word2vec_features.reshape(1, -1)], axis=1)
    return combined_features

if __name__=='__main__':
    app.run(debug=True)