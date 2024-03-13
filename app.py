from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from langchain import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from werkzeug.utils import secure_filename
import os
import secrets


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database file
app.config['SECRET_KEY'] = secrets.token_hex(16)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    summarizations = db.relationship('SummarizationHistory', backref='user', lazy=True)

class SummarizationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    summarized_text = db.Column(db.Text, nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home')
def home():
      # Check if the user is logged in
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.filter_by(id=user_id).first()

        # Retrieve the summarization history for the logged-in user
        summarization_history = SummarizationHistory.query.filter_by(user_id=user.id).all()

        return render_template('mainwebpage.html', summarization_history=summarization_history)

    # If the user is not logged in, redirect them to the login page
    flash('Please log in to access the home page.', 'warning')
    return redirect(url_for('login'))

#The definition of the LLM used for summarization 
summarizer_llm = HuggingFaceHub(
    repo_id="Falconsai/text_summarization",
    model_kwargs={"temperature":0, "max_length":1000, "min_length": 30}
)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        session['user_id'] = user.id
 
        if user and bcrypt.check_password_hash(user.password, password):
            # Login successful
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check your username and password.', 'danger')

    return render_template('login.html')


@app.route('/summarize', methods=['POST'])
def summarize() -> str:
    user_input = request.form['userText']
    summary_result = summarizer_llm(f"Summarize this text without omitting any specific details: {user_input}!")
    # Store the summarization in the database
    if 'user_id' in session:
        user_id = session['user_id']
        new_summarization = SummarizationHistory(user_id=user_id, input_text=user_input, summarized_text=summary_result)
        db.session.add(new_summarization)
        db.session.commit()

    return render_template('result.html', user_input=user_input, summary=summary_result)

@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf() -> str:
    # Check if the request contains a file
    if 'pdf_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    file = request.files['pdf_file']

    # Check if the user submitted an empty file input
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    # Check if the file is a PDF
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        loader = PyPDFLoader(filepath)  # Use the uploaded file path
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        db_faiss = FAISS.from_documents(docs, embeddings)
        qa = RetrievalQA.from_chain_type(llm=summarizer_llm, retriever=db_faiss.as_retriever())      
        ans = qa({'query': 'Summarize the whole document.'})['result']

        if 'user_id' in session:
            user_id = session['user_id']
            new_summarization = SummarizationHistory(user_id=user_id, input_text=filename, summarized_text=ans)
            db.session.add(new_summarization)
            db.session.commit()

        return render_template('result_pdf.html', filename = filename, summary=ans)  # Update the template accordingly
      
    else:
        flash('Invalid file format. Please upload a PDF file.', 'danger')





if __name__ == '__main__':
    app.run(debug=True)

