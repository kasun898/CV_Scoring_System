import fitz  # PyMuPDF
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Function to calculate score
def calculate_score(cv_tokens, job_advert_tokens):
    # Create a Counter object for job advert tokens
    job_advert_counter = Counter(job_advert_tokens)
    # Create a Counter object for CV tokens
    cv_counter = Counter(cv_tokens)
    # Calculate the matching score
    score = sum(cv_counter[token] * job_advert_counter[token] for token in job_advert_counter)
    return score

# Extract text from PDFs
job_advert_text = extract_text_from_pdf('job_advert.pdf')
cv_text = extract_text_from_pdf('CV.pdf')

# Preprocess the text
job_advert_tokens = preprocess_text(job_advert_text)
cv_tokens = preprocess_text(cv_text)

# Calculate the score
score = calculate_score(cv_tokens, job_advert_tokens)

print(f"Matching Score: {score}")

# Optional: Normalize the score
max_score = sum(Counter(job_advert_tokens).values())
normalized_score = score / max_score * 100

print(f"Normalized Matching Score: {normalized_score:.2f}%")
