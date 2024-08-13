import pickle
import string
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, render_template, jsonify
from langdetect import detect
from googletrans import Translator

# Initialize NLTK
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
if not len(firebase_admin._apps):
    cred = credentials.Certificate("config.json")
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Load the original dataset
try:
    original_data = pd.read_csv('spam.csv', encoding='utf-8')
except UnicodeDecodeError:
    original_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Rename columns if necessary
original_data = original_data.rename(columns={'v2': 'message', 'v1': 'label'})

# Initialize translator
translator = Translator()

# Language Detection Function
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'unknown'

# Translation Function
def translate_to_english(text, src_lang):
    try:
        if src_lang != 'en':
            translated = translator.translate(text, src=src_lang, dest='en')
            return translated.text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return the original text if there's an error

# Preprocessing function
def dealing_with_txt(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Check if the message is in the dataset
def is_message_in_dataset(message, dataset):
    preprocessed_message = dealing_with_txt(message)
    return preprocessed_message in dataset['message'].apply(dealing_with_txt).values

# Load Firebase data
def load_firebase_data():
    firebase_data = db.collection('feedback').stream()
    firebase_messages = [doc.to_dict() for doc in firebase_data]
    return pd.DataFrame(firebase_messages)

# Handle new data and feedback
def handle_new_data(input_sms, prediction, user_feedback, corrected_label=None):
    feedback_ref = db.collection('feedback')
    
    # Determine the final label to store
    if user_feedback == 'No' and corrected_label:
        final_label = corrected_label
    else:
        final_label = prediction

    feedback_doc = {
        'message': input_sms,
        'label': final_label
    }
    if user_feedback:
        feedback_doc['user_feedback'] = user_feedback
    feedback_ref.add(feedback_doc)

# Train new model on Firebase data
def train_new_model(firebase_data):
    # Preprocess the data
    firebase_data['processed_message'] = firebase_data['message'].apply(dealing_with_txt)
    X_train = tfidf.transform(firebase_data['processed_message'])
    y_train = firebase_data['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Train a new model
    from sklearn.naive_bayes import MultinomialNB
    new_model = MultinomialNB()
    new_model.fit(X_train, y_train)
    
    print("Training on Firebase data done.")
    return new_model

# Retrieve and train new model on Firebase data at the start
firebase_data = load_firebase_data()
new_model = train_new_model(firebase_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['message']
    
    # Check if the message is in the combined dataset
    combined_data = pd.concat([
        original_data[['message', 'label']],
        firebase_data
    ], ignore_index=True)
    
    if is_message_in_dataset(input_sms, combined_data):
        preprocessed_message = dealing_with_txt(input_sms)
        vector_input = tfidf.transform([preprocessed_message])
        
        # Predict with both models
        original_prediction = model.predict(vector_input)[0]
        original_label = "spam" if original_prediction == 1 else "ham"
        
        if new_model:
            new_prediction = new_model.predict(vector_input)[0]
            new_label = "spam" if new_prediction == 1 else "ham"
        else:
            new_label = original_label  # Use original label if no new model is available
        
        # Combine predictions (majority vote or another strategy)
        combined_label = new_label if new_label == original_label else "spam"  # Simplified strategy
        
        return render_template('predict.html', message=input_sms, prediction=combined_label, is_in_dataset=True)
    else:
        # Predict with original model
        transformed_sms = dealing_with_txt(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        
        original_prediction = model.predict(vector_input)[0]
        original_label = "spam" if original_prediction == 1 else "ham"
        
        # Predict with new model
        if new_model:
            new_prediction = new_model.predict(vector_input)[0]
            new_label = "spam" if new_prediction == 1 else "ham"
        else:
            new_label = original_label  # Use original label if no new model is available
        
        # Combine predictions
        combined_label = new_label if new_label == original_label else "spam"  # Simplified strategy
        
        # Display result and ask for feedback
        return render_template('predict.html', message=input_sms, prediction=combined_label, is_in_dataset=False)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    message = request.form.get('message')
    prediction = request.form.get('prediction')
    user_feedback = request.form.get('user_feedback')
    corrected_label = request.form.get('corrected_label')

    handle_new_data(message, prediction, user_feedback, corrected_label)

    return jsonify({'status': 'Feedback submitted successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
