import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import sklearn
from functools import reduce
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    return " ".join(y)


def remove_pattern(input_txt,pattern):
    input_txt=input_txt.lower()
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
    return input_txt


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Tweet Recognition")

input_txt = st.text_area("Enter the Tweet")

if st.button('Predict'):

    # 1. preprocess
    input_txt=remove_pattern(input_txt,"@[\w]*")
    input_txt = re.sub("^[a-zA-Z#]", " ",input_txt)
    input_txt=' '.join([w for w in input_txt.split() if(len(w) > 3)])
    input_txt=transform_text(input_txt)
    words=word_tokenize(input_txt)
    stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")

    # 2. vectorizer
    vector_input=tfidf.transform([stemmed_sentence])

    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    print(result)
    if result == 1:
        st.header("Racist/Hated Tweet")
    else:
        st.header("Non-Racisit/Positive Tweet")