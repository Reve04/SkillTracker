import os
import io
import time
import json
import re
import logging
import traceback
import concurrent.futures
import spacy
import plotly
import plotly.express as px
import pandas as pd
import uuid
import pytesseract
from functools import lru_cache
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, login_required, 
                        logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Flask-Login configuration
login_manager = LoginManager(app)
login_manager.login_view = 'login'
nlp = spacy.load("en_core_web_sm")
db = SQLAlchemy(app)

# Configuration constants for file processing
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TESSERACT_CONFIG = r'--oem 3 --psm 6'  # Optimized OCR settings

# -------------------------------------------
# Database Model
# -------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    skills = db.Column(db.JSON, default=list)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------------------
# NLP Model & Skill Matching Functions
# -------------------------------------------
@lru_cache(maxsize=1)
def get_nlp_model():
    """Load and cache the spaCy NLP model for faster performance."""
    return spacy.load("en_core_web_sm")

def load_skill_patterns():
    """Pre-load a list of skill patterns for efficient matching."""
    skill_keywords = [
        "Python", "Machine Learning", "AI", "JavaScript", "React",
        "Data Analysis", "SQL", "TensorFlow", "PyTorch", "Flask",
        "Java", "AWS", "Docker", "Git", "HTML", "CSS", "Node.js",
        "Deep Learning", "Statistics", "Mathematics", "Django Framework",
        "Project Management"
    ]
    nlp_model = get_nlp_model()
    return [nlp_model.make_doc(skill.lower()) for skill in skill_keywords]

def detect_skills(text):
    try:
        if not text:
            return []
            
        # Normalize text and skills
        text = re.sub(r'\W+', ' ', text.lower())
        nlp = get_nlp_model()
        doc = nlp(text)
        
        # Use exact skill matching with normalization
        skill_patterns = load_skill_patterns()
        matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
        matcher.add("SKILLS", skill_patterns)
        matches = matcher(doc)
        
        skills = set()
        for match_id, start, end in matches:
            skill = doc[start:end].text.title()
            skills.add(skill)
            
        return list(skills)
    except Exception as e:
        traceback.print_exc()
        return []

# -------------------------------------------
# File Extraction Functions
# -------------------------------------------
def extract_text_from_file(file):
    """
    Validate and extract text from an uploaded file.
    Supports PDF and image files. Uses parallel processing for PDFs.
    """
    try:
        # Read file into memory
        file_bytes = io.BytesIO(file.read())
        # Validate file size
        file_bytes.seek(0, os.SEEK_END)
        file_size = file_bytes.tell()
        file_bytes.seek(0)
        if file_size > MAX_FILE_SIZE:
            raise ValueError("File size exceeds 5MB limit")
        # Process PDF
        if file.filename.lower().endswith('.pdf'):
            return process_pdf(file_bytes)
        # Process image files
        elif file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            return process_image(file_bytes)
        else:
            return ""
    except Exception as e:
        logging.error(f"Extraction error: {str(e)}")
        return ""

def process_pdf(file_bytes):
    """Process PDF files with parallel page extraction."""
    text = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        reader = PdfReader(file_bytes)
        if reader.is_encrypted:
            reader.decrypt('')
        futures = []
        for page in reader.pages:
            futures.append(executor.submit(extract_page_text, page))
        for future in concurrent.futures.as_completed(futures):
            text.append(future.result())
    return " ".join(text).strip()

def extract_page_text(page):
    """Extract text from a single PDF page."""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""

def process_image(file_bytes):
    """Process image files with pre-processing steps for OCR."""
    img = Image.open(file_bytes)
    # Convert to grayscale for faster OCR
    img = img.convert('L')
    # Resize if too large while maintaining aspect ratio
    max_size = 2000
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    return pytesseract.image_to_string(img, config=TESSERACT_CONFIG).strip()

# -------------------------------------------
# Job Recommendation Function
# -------------------------------------------
def get_job_recommendations(skills):
    """
    Provide job recommendations based on the user's skills.
    Uses TF-IDF vectorization and cosine similarity to score jobs.
    """
    jobs = [
        {
            "title": "Machine Learning Engineer",
            "company": "AI Innovations",
            "skills": ["Python", "Machine Learning", "TensorFlow", "Deep Learning"],
            "location": "Remote",
            "salary": "$120k - $150k",
            "type": "Full-time"
        },
        {
            "title": "Senior Web Developer",
            "company": "WebTech Solutions",
            "skills": ["JavaScript", "React", "Node.js", "AWS"],
            "location": "New York",
            "salary": "$90k - $130k",
            "type": "Contract"
        },
        {
            "title": "Data Scientist",
            "company": "Data Insights Co.",
            "skills": ["Python", "SQL", "Statistics", "Machine Learning"],
            "location": "London",
            "salary": "£60k - £80k",
            "type": "Full-time"
        }
    ]
    
    # Normalize skills for comparison
    user_skills = set(skill.lower() for skill in skills)
    
    # Calculate matches using exact skill comparison
    for job in jobs:
        job_skills = set(skill.lower() for skill in job['skills'])
        matched_skills = user_skills & job_skills
        match_percentage = int((len(matched_skills) / len(job_skills)) * 100) if job_skills else 0
        job['match'] = min(max(match_percentage, 40), 95)  # Keep between 40-95%
        job['matched_skills'] = [skill for skill in job['skills'] if skill.lower() in user_skills]
    
    return sorted(jobs, key=lambda x: x['match'], reverse=True)

# -------------------------------------------
# Routes
# -------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    skills = current_user.skills if current_user.skills else []
    
    if skills:
        skill_counts = {skill: 1 for skill in skills}
        df = pd.DataFrame({
            'Skill': list(skill_counts.keys()),
            'Count': list(skill_counts.values())
        })
        fig = px.pie(
            df, names='Skill', values='Count',
            title="Your Skill Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            margin=dict(t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial'
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph_json = None
    return render_template('dashboard.html', graph_json=graph_json)

@app.route('/upload', methods=['POST'])
@login_required
def upload_certificate():
    if 'certificate' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['certificate']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Extract text using the optimized function
        text = extract_text_from_file(file)
        if not text:
            raise ValueError("Could not extract text from file")
        # Detect skills using the optimized NLP function
        new_skills = detect_skills(text)
        if not new_skills:
            raise ValueError("No skills detected in the document")
        # Update user skills (ensuring uniqueness)
        existing_skills = current_user.skills if current_user.skills else []
        updated_skills = list(set(existing_skills + new_skills))
        current_user.skills = updated_skills
        db.session.commit()
        flash(f'Added {len(new_skills)} new skills!', 'success')
    except Exception as e:
        traceback.print_exc()
        flash(str(e), 'danger')
    return redirect(url_for('dashboard'))

@app.route('/delete_skills', methods=['POST'])
@login_required
def delete_skills():
    current_user.skills = []
    db.session.commit()
    flash('All skills have been reset!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/jobs')
@login_required
def job_recommendations():
    if not current_user.skills:
        flash('Upload certificates to get job recommendations', 'warning')
        return redirect(url_for('dashboard'))
        
    jobs = get_job_recommendations(current_user.skills)
    current_user_skills = [skill.lower() for skill in current_user.skills]
    return render_template('jobs.html', 
                         jobs=jobs,
                         current_user_skills=current_user_skills)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# -------------------------------------------
# Database Initialization
# -------------------------------------------
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)