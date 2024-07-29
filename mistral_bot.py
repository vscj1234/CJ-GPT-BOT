import re
from flask import Flask, render_template, request, jsonify, session
from flask_mail import Mail, Message
import atexit
import mysql.connector as mysql
from langchain_community.llms import HuggingFaceHub
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
import os
import chromadb
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from flask_session import Session

class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        # Implement here your streaming logic
        print(token, end='', flush=True)

callback_manager = CallbackManager(
    [CustomCallbackHandler(), StreamingStdOutCallbackHandler()]
)
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_new_tokens": 512})

# Adding the documents from the disk
documents = SimpleDirectoryReader("./data").load_data()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("cloudjune")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
query_engine = index.as_query_engine()

# MySQL Database Configuration for Conversations
conversation_db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'uch$xilC2g', #please change ur password
    'database': 'cloudjunebot',
}


conversation_connection = mysql.connect(**conversation_db_config)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "3db1083617e63bb0f7d9fbda2020cd8e"  # Change this to a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize SendGrid API key
SENDGRID_API_KEY = "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"
os.environ["SENDGRID_API_KEY"] = SENDGRID_API_KEY

# Define SendGrid email sending function
def send_email(user_message, bot_response):
    message = Mail(
        from_email='support@cloudjune.com',
        to_emails='marketing@cloudjune.com',  # Update with recipient email
        subject='User Enquiry',
        html_content=f'<p>User Message: {user_message}</p><p>Bot Response: {bot_response}</p>')

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))

# Define routes
@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize conversation history if not present in session
    if 'conversation_history' not in session:
        session['conversation_history'] = [
            "You are a pro conversational chatbot created for a company named CloudJune, you are pro at understanding context and deliver meaningful conversational responses. Your responses should solely depend on the custom data you are trained about the company. You should always refuse to answer questions that are not related to CloudJune. After every response you should ask the user to share their name and email so that the team can reach out to them. Do it only until they provide their name and email. Act accordingly:"
        ]

    message = request.json["message"]
    
    # Append user's message to the conversation history
    session['conversation_history'].append(message)
    
    # Use the entire conversation history as context for generating responses
    context = " ".join(session['conversation_history'])
    response = query_engine.query(context)  
    response = str(response).split("Answer:", 1)[1]
    
    # Update conversation history with the assistant's response
    session['conversation_history'].append(response)
    context = " ".join(session['conversation_history'])
    print("Question:", message)
    print("Answer:", response)

    # Check if the user message contains an email address
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, message)
    
    # If an email address is found, send an email using SendGrid
    if emails_found:
        send_email(message, response)

    # Insert user query into conversations table
    cursor = conversation_connection.cursor()
    user_id = session.sid  # Using session ID as user identifier
    cursor.execute('INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)', (user_id, message, response))
    conversation_connection.commit()
    cursor.close()

    return jsonify({"answer": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

@atexit.register
def on_exit():
    conversation_connection.close()
