from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import atexit
import mysql.connector as mysql
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import re
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize SendGrid API key
SENDGRID_API_KEY = "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"
os.environ["SENDGRID_API_KEY"] = SENDGRID_API_KEY

# MySQL Database Configuration for Conversations and Appointments
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': ' your password', 
    'database': 'cloudjune',
}

db_connection = mysql.connect(**db_config)

os.environ["OPENAI_API_KEY"] = "your api key"

# Load the persisted vector store
embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(model="gpt-4")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

# Define SendGrid email sending function
def send_email(user_message, bot_response):
    message = Mail(
        from_email='support@cloudjune.com',
        to_emails='marketing@cloudjune.com',  # Update with recipient email
        subject='User Enquiry',
        html_content=f'<p>User Message: {user_message}</p><p>Bot Response: {bot_response}</p>'
    )

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))

@app.route('/')
def home():
    return render_template('base.html')

# Function to check available appointment slots
def get_available_slots():
    cursor = db_connection.cursor()
    cursor.execute("SELECT appointment_time FROM appointments")
    booked_slots = [row[0] for row in cursor.fetchall()]
    cursor.close()

    available_slots = []
    now = datetime.now()
    for i in range(7):  # Next 7 days
        date = now.date() + timedelta(days=i)
        for hour in range(9, 18):  # 9 AM to 5 PM
            slot = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            if slot not in booked_slots:
                available_slots.append(slot.strftime("%Y-%m-%d %H:00"))

    return available_slots[:5]  # Return the first 5 available slots

# Function to book an appointment
def book_appointment(user_id, appointment_time):
    cursor = db_connection.cursor()
    cursor.execute("INSERT INTO appointments (user_id, appointment_time) VALUES (%s, %s)", (user_id, appointment_time))
    db_connection.commit()
    cursor.close()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data['message']

    # Initialize chat history if not present in session
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'booking_stage' not in session:
        session['booking_stage'] = 'initial'

    # Prepare the conversation history
    chat_history = session['chat_history']

    # Add system message if it's the first message in the conversation
    if not chat_history:
        system_message = ("You are a pro conversational chatbot created for a company named CloudJune. You should act as a chat assistant on behalf of CloudJune and should always refuse to answer questions that are not related to CloudJune. "
                          "You are pro at understanding context and deliver meaningful conversational responses. "
                          "You have the ability to schedule appointments for users. When a user expresses interest in booking an appointment, guide them through the process using the available appointment slots. "
                          "After every response you should ask the user to share their name and email so that the team can reach out to them. "
                          "Do it only until they provide their name and email. Act accordingly:")
        chat_history.append(("system", system_message))

    # Handle appointment booking process
    if session['booking_stage'] == 'initial' and any(keyword in query.lower() for keyword in ["book", "appointment", "schedule"]):
        session['booking_stage'] = 'started'
        available_slots = get_available_slots()
        response = "Certainly! I'd be happy to help you book an appointment with CloudJune. Here are the available slots:\n\n"
        for i, slot in enumerate(available_slots, 1):
            response += f"{i}. {slot}\n"
        response += "\nPlease choose a slot by entering its number."
    elif session['booking_stage'] == 'started':
        try:
            slot_index = int(query) - 1
            available_slots = get_available_slots()
            chosen_slot = available_slots[slot_index]
            book_appointment(session.sid, chosen_slot)
            response = f"Great! Your appointment with CloudJune has been booked for {chosen_slot}. Is there anything else I can help you with?"
            session['booking_stage'] = 'initial'
        except (ValueError, IndexError):
            response = "I'm sorry, that's not a valid selection. Please choose a number from the list of available slots."
    else:
        # Generate response using the existing chain
        result = chain({"question": query, "chat_history": chat_history})
        response = result['answer']

        # Check if the response indicates inability to book appointments
        if "don't have the ability to schedule appointments" in response:
            response = "I apologize for the confusion. I can actually help you book an appointment with CloudJune. Would you like me to show you the available slots?"
            session['booking_stage'] = 'initial'

    # Update chat history
    chat_history.append(("user", query))
    chat_history.append(("bot", response))

    # Trim chat history if it gets too long (keep last 10 exchanges)
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # Save updated chat history to session
    session['chat_history'] = chat_history

    # Check if the user message contains an email address
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, query)

    # If an email address is found, send an email using SendGrid
    if emails_found:
        send_email(query, response)

    # Insert user query into conversations table
    cursor = db_connection.cursor()
    user_id = session.sid  # Using session ID as user identifier
    cursor.execute('INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)', (user_id, query, response))
    db_connection.commit()
    cursor.close()

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

@atexit.register
def on_exit():
    if db_connection.is_connected():
        db_connection.close()
