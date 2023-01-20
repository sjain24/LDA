import pickle


def loadModel():
    lda_model_file = open("lda_model_auto_5.pkl", "rb")
    lda_model = pickle.load(lda_model_file)
    lda_model_file.close()
    return lda_model


def run_inference(text):
    lda_model = loadModel()
    words = lda_model.show_topics(num_topics=5, num_words=20, log=False, formatted=True)
    words = [value for item, value in words]
    return lda_model[text], words
