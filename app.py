import os
import sqlite3
import pickle
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, g
from flask_bcrypt import Bcrypt
import re

# --- Initialize the Flask App and Bcrypt ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Set a secret key for sessions
bcrypt = Bcrypt(app)

# --- Configuration ---
DATABASE = 'users.db'
MODEL_FILE = 'spam_detector_pipeline.pkl'
PHISHING_MODEL_FILE = 'phishing_model.pkl'
PHISHING_VECTORIZER_FILE = 'phishing_vectorizer.pkl'

# --- Database Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # Allows accessing columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        db.commit()

# --- Model Loading ---
model_pipeline = None
phishing_model = None
phishing_vectorizer = None

try:
    with open(MODEL_FILE, 'rb') as model_file:
        model_pipeline = pickle.load(model_file)
except FileNotFoundError:
    print(f"Warning: The model file '{MODEL_FILE}' was not found.")
    print("SMS/Email spam detection functionality will not work.")

try:
    phishing_model = joblib.load(PHISHING_MODEL_FILE)
    phishing_vectorizer = joblib.load(PHISHING_VECTORIZER_FILE)
except FileNotFoundError:
    print(f"Warning: The phishing model files '{PHISHING_MODEL_FILE}' and '{PHISHING_VECTORIZER_FILE}' were not found.")
    print("Phishing detection functionality will not work.")

# --- Routes ---

# Home Page
@app.route('/')
def home():
    return render_template('index.html', logged_in='logged_in' in session, username=session.get('username'))

# About Project Page
@app.route('/about')
def about():
    return render_template('about.html', logged_in='logged_in' in session, username=session.get('username'))

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match.", logged_in=False)
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists.", logged_in=False)
    
    return render_template('register.html', logged_in=False)

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        cursor = db.cursor()
        user = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user and bcrypt.check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password.", logged_in=False)
    
    return render_template('login.html', logged_in=False)

# Logout Route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('home'))

# SMS Detection Page
@app.route('/sms_detection', methods=['GET', 'POST'])
def sms_detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    result = None
    if request.method == 'POST':
        text_content = request.form['text_content']
        if model_pipeline:
            # The model expects a list of strings
            prediction = model_pipeline.predict([text_content])
            if prediction[0] == 1:
                result = "Spam"
            else:
                result = "Not Spam (Ham)"
        else:
            result = "Error: Model not loaded."
    
    return render_template('sms_detection.html', result=result, logged_in='logged_in' in session, username=session.get('username'))

# Email Detection Page
@app.route('/email_detection', methods=['GET', 'POST'])
def email_detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    result = None
    if request.method == 'POST':
        text_content = request.form['text_content']
        if model_pipeline:
            # The model expects a list of strings
            prediction = model_pipeline.predict([text_content])
            if prediction[0] == 1:
                result = "Spam"
            else:
                result = "Not Spam (Ham)"
        else:
            result = "Error: Model not loaded."
    
    return render_template('email_detection.html', result=result, logged_in='logged_in' in session, username=session.get('username'))

# Phishing URL/Domain Detection Page
def preprocess_url(url):
    """Performs basic preprocessing on the URL."""
    url = re.sub(r'https?://(?:www\.)?', '', url)
    return url

@app.route('/phishing_detection', methods=['GET', 'POST'])
def phishing_detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        url_input = request.form['url_input']
        if phishing_model and phishing_vectorizer:
            # Preprocess the URL and transform it using the vectorizer
            preprocessed_url = preprocess_url(url_input)
            url_vectorized = phishing_vectorizer.transform([preprocessed_url])
            
            # Make the prediction
            prediction = phishing_model.predict(url_vectorized)
            
            if prediction[0] == 1:
                result = "Phishing"
            else:
                result = "Legitimate"
        else:
            result = "Error: Phishing model not loaded."

    return render_template('phishing_detection.html', result=result, logged_in='logged_in' in session, username=session.get('username'))

# --- Main entry point ---
if __name__ == '__main__':
    init_db() # Initialize the database
    app.run(debug=True)
