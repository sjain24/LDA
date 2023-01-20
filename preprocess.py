from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle
import streamlit as st

# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

new_stop_words = ['ooh', 'yeah', 'hey', 'whoa', 'woah', 'ohh', 'was', 'mmm', 'oooh', 'yah', 'yeh', 'mmm', 'hmm', 'deh',
                  'doh', 'jah', 'wa', 'ohhh']


def preProcess(text):
    if not checkIfEnglish(text):
        raise Exception
    tokenized_text = tokenize(text)
    tokenized_text = lemmetize(tokenized_text)
    tokenized_text = removeStopWords(tokenized_text)
    dictionary = loadDictionary()
    return dictionary.doc2bow(tokenized_text), tokenized_text


def checkIfEnglish(text):
    try:
        langlist = detect_langs(text)
        for l in langlist:
            if l.lang != 'en' or l.prob < 0.95:  # lyrics not clearly identified as English
                continue
            else:
                return True
    except LangDetectException:
        return False
    return False


def tokenize(song):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_lyric = tokenizer.tokenize(song.lower())
    filtered_song = []
    for token in tokenized_lyric:
        if len(token) > 2 and not token.isnumeric():
            filtered_song.append(token)
    return filtered_song


def lemmetize(song):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in song:
        try:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
        except Exception as e:
            print(e)
    return lemmatized_tokens


def removeStopWords(song):
    stop_words = stopwords.words('english')
    stop_words.extend(new_stop_words)

    filtered_text = []
    for token in song:
        if token not in stop_words:
            filtered_text.append(token)
    return filtered_text


@st.cache
def loadDictionary():
    dictionary_file = open("dictionary.pkl", "rb")
    dictionary = pickle.load(dictionary_file)
    dictionary_file.close()
    return dictionary