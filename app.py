from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim.downloader as api

# Load the model and label encoder
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# print(model)
label_encoder = ["negative", "neutral", "positive"]

# Load word vectors
# wv = api.load('glove-wiki-gigaword-50')
wv=api.load('word2vec-google-news-300')

# NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def get_vector(words):
    vectors = [wv[word] for word in words if word in wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(wv.vector_size)

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    if request.method == "POST":
        text = request.form["review"]
        processed = preprocess(text)
        vec = np.array([get_vector(processed)])
        prediction = model.predict(vec)
        print(prediction)
        sentiment = label_encoder[prediction[0]]
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
