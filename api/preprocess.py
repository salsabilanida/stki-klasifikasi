from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess(text):
    # Case Folding
    # menghapus garis miring dan tanda kutip
    text = re.sub("\'", "", str(text))
    # menghapus semuanya kecuali huruf
    text = re.sub("[^a-zA-Z]", " ", text)
    # menghapus spasi
    text = ' '.join(text.split())
    # mengubah text menjadi huruf kecil
    text = text.lower()

    # Tokenizing
    tokens = word_tokenize(text)

    # Filtering/Penghapusan stopwords
    stop_words = set(stopwords.words('indonesian', 'english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Penggabungan kembali token menjadi teks
    processed_text = ' '.join(tokens)
    return processed_text
