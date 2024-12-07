import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langdetect import detect
from google.cloud import translate_v2 as translate
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TensorFlow model
model = tf.saved_model.load("saved_model")

# Load class labels
with open("class_indices.json", "r") as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# Set up Groq Chatbot and Prompt
GROQ_API_KEY = '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'
GOOGLE_API_KEY = 'AIzaSyBkoorRTaH08H3RFIft4ug6bT1ABexXswI'
GOOGLE_TRANS_KEY = 'steam-lock-443918-p0-d8556e5bba5d.json'
GOOGLE_STT_KEY = 'speech-to-text-443918-bc8f683f7dd2.json'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

translate_client = translate.Client.from_service_account_json(GOOGLE_TRANS_KEY)

def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("rag_docs")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

vectors = vector_embedding()
retriever = vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    input_tensor = tf.convert_to_tensor(input_data)

    infer = model.signatures['serving_default']
    output_data = infer(input_tensor)
    logits = list(output_data.values())[0].numpy()[0]
    return logits

def detect_language(text):
    return detect(text)

def translate_to_english(text, target_lang='en'):
    if detect(text) != 'en':
        translated = translate_client.translate(text, target_language=target_lang)
        return translated['translatedText']
    return text

def translate_from_english(text, target_lang='en'):
    if target_lang != 'en':
        translated = translate_client.translate(text, target_language=target_lang)
        return translated['translatedText']
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            unique_filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            session['uploaded_image'] = unique_filename
            session['chat_history'] = []

            logits = process_image(file_path)
            predicted_index = np.argmax(logits)
            predicted_label = class_labels[predicted_index]
            session['model_output'] = predicted_label

            return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results')
def results():
    uploaded_image = session.get('uploaded_image', None)
    model_output = session.get('model_output', None)
    chat_history = session.get('chat_history', [])
    return render_template('results.html',
                           uploaded_image=uploaded_image,
                           model_output=model_output,
                           chat_history=chat_history)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    chat_history = session.get('chat_history', [])

    if user_input:
        try:
            detected_lang = detect_language(user_input)
            user_input_in_english = translate_to_english(user_input)
            prompt_text = f"Answer the following question in English:\n{user_input_in_english}"
            response = retrieval_chain.invoke({'input': prompt_text})
            raw_response = response.get('answer', 'No response available')
            translated_response = translate_from_english(raw_response, target_lang=detected_lang)
            chat_history.append({'user': user_input, 'response': translated_response})
            session['chat_history'] = chat_history
            return jsonify({'user': user_input, 'bot': translated_response})
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            return jsonify({'error': 'An error occurred while processing your request.'})
    return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
