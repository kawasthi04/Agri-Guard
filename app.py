import os
import json
import uuid
import time
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
from langdetect import detect  # Language detection library
from google.cloud import translate_v2 as translate  # Google Translate API

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model from the .pb file
model = tf.saved_model.load("saved_model")

# Load class labels from a JSON file
with open("class_indices.json", "r") as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# Set up Groq Chatbot
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

# Initialize Google Translate client
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
    """
    Detect the language of the input text.
    Returns the language code (e.g., 'en', 'hi', 'ta')
    """
    return detect(text)

def translate_to_english(text, target_lang='en'):
    """
    Translate the input text to English.
    Uses Google Translate API.
    """
    if detect(text) != 'en':  # Only translate if the language is not English
        translated = translate_client.translate(text, target_language=target_lang)
        return translated['translatedText']
    return text

def translate_from_english(text, target_lang='en'):
    """
    Translate the input text from English to the target language.
    """
    if target_lang != 'en':
        translated = translate_client.translate(text, target_language=target_lang)
        return translated['translatedText']
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Create a unique filename and save the image
            unique_filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Log for debugging
            print(f"Image saved at: {file_path}")

            # Store the image and clear the chat history
            session['uploaded_image'] = unique_filename
            session['chat_history'] = []

            # Process the image and get the predicted label
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
        # Detect the language of the user's input
        detected_lang = detect_language(user_input)

        # Translate the input to English
        user_input_in_english = translate_to_english(user_input)

        # Construct the prompt for the LLM in English
        prompt_text = f"Answer the following question in English:\n{user_input_in_english}"

        # Pass the prompt to the model for processing
        response = retrieval_chain.invoke({'input': prompt_text})
        raw_response = response.get('answer', 'No response available')

        # Translate the response back to the user's language
        translated_response = translate_from_english(raw_response, target_lang=detected_lang)

        # Format the response
        formatted_response = translated_response.replace('\n', '').replace('* ', '\n').split('\n')
        formatted_response = [item.strip() for item in formatted_response if item]

        structured_response = """
        <div>
            <h2>AgriGuard says:</h2>
            <br/>
            <ol>
        """
        for i, item in enumerate(formatted_response, start=1):
            structured_response += f"<li>{item}</li>"
        structured_response += """
            </ol>
            <br/>
            <p><em>Need more details? Feel free to ask more questions!</em></p>
        </div>
        """

        # Log and save chat history
        chat_history.append(f"User ({detected_lang}): {user_input}")
        chat_history.append({'type': 'bot', 'response': structured_response})
        session['chat_history'] = chat_history

        return jsonify({'user': user_input, 'bot': structured_response})

    return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)