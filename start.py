import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()


def dealing_with_txt(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

#tfidf = pickle.load(open(r"C:\Users\user\Desktop\ham" or "spam\vectorizer.pkl", 'rb'))
tfidf = pickle.load(open(r"vectorizer.pkl", 'rb'))

#model = pickle.load(open(r"C:\Users\user\Desktop\ham or spam\model.pkl", 'rb'))
model = pickle.load(open(r"model.pkl", 'rb'))
st.markdown("<h6 style='text-align: center'>Welcome to the App!💗💗</h1>", unsafe_allow_html=True)

st.title("Ham or Spam Classifier (Email/SMS)")
st.image("img.jpeg" ,use_column_width=True)

st.text("""
This app uses a ML classifier to predict whether an email or SMS message is spam or 
ham (not spam). To use the app, enter your message in the text box below and click 
the 'Predict' button.""")



input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = dealing_with_txt(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("The Message / Email is Spam🔴")
    else:
        st.header("The Message / Email is Ham/Not Spam🟢")