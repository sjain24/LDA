import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from preprocess import preProcess
from run import run_inference
import re

matplotlib.use("agg")
_lock = RendererAgg.lock

labels = ["Obstacles & Time", "Romance", "Religious", "Violence/Explicit", "Nature & Home"]


def showChart(list_of_confidence_scores):
    data = np.zeros(len(labels))
    for score in list_of_confidence_scores:
        data[score[0]] = score[1]

    plt.figure(figsize = (10, 5))
    st.markdown("""---""")
    st.subheader("Most suitable theme for the song is [" + labels[np.argmax(data)] + "]")
    st.markdown("""---""")
    # creating the bar plot
    plt.bar(labels, data, color='blue', width=0.4)

    plt.ylabel("Confidence Scores")
    plt.title("Theme Analysis from Song Lyrics")
    st.pyplot(plt)
    # return index which has max score
    return np.argmax(data)


def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw


def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


st.title('Song Theme Analysis from Lyrics')

text = st.text_area(label="Enter lyrics here")


def explainResponse(topic_words_list, lyric_words):
    topic_words_list = re.findall('"([^"]*)"', topic_words_list)
    res = []
    for x in lyric_words:
        if x not in res:
            res.append(x)
    lyric_words = res
    lyric_words_vec = [word2vec(word) for word in lyric_words]
    topic_words_vec = [word2vec(word) for word in topic_words_list]
    cosine_score = [-1] * len(lyric_words)
    for idx, lyric_word in enumerate(lyric_words_vec):
        max_score = -1
        for topic_word in topic_words_vec:
            max_score = max(cosdis(lyric_word, topic_word), max_score)
        cosine_score[idx] = max_score
    max_val = max(cosine_score)
    top_words = [lyric_words[i] for i, value in enumerate(cosine_score) if value == max_val]
    st.markdown("""---""")
    st.subheader("Top 3 words from the lyrics which matched this topic are ["
                 + ", ".join([str(elem) for elem in top_words[:3]]) + "]")


if st.button('Analyze it!'):
    try:
        text, text_tokenized = preProcess(text)
        response, topic_words = run_inference(text)
        index = showChart(response)
        explainResponse(topic_words[index], text_tokenized)
    except Exception as e:
        st.subheader("Something went wrong while analysing this text! :" + e)

