import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extract text from the sample PDFs
kasun_text = extract_text_from_pdf('Kasun Amarasinge CV.pdf')
ranasinghe_text = extract_text_from_pdf('ranasinghe.pdf')
thakshila_text = extract_text_from_pdf('thakshila.pdf')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

# Preprocess the extracted text
kasun_cleaned = preprocess_text(kasun_text)
ranasinghe_cleaned = preprocess_text(ranasinghe_text)
thakshila_cleaned = preprocess_text(thakshila_text)

# Convert the preprocessed text into numerical features using TF-IDF
documents = [kasun_cleaned, ranasinghe_cleaned, thakshila_cleaned]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

def score_cv(text):
    score = 0
    max_score = 5  
    if 'finance' in text:
        score += 1
    if 'credit' in text:
        score += 1
    if 'analysis' in text:
        score += 1
    if 'cfa' in text or 'cpa' in text:
        score += 1
    if 'data analysis' in text:
        score += 1
    percentage_score = (score / max_score) * 100  # Convert score to percentage
    return percentage_score

# Score each CV
kasun_score = score_cv(kasun_cleaned)
ranasinghe_score = score_cv(ranasinghe_cleaned)
thakshila_score = score_cv(thakshila_cleaned)

# Create a data table 
data = {
    'Candidate': ['Kasun Amarasinghe', 'T.D. Ranasinghe', 'Thakshila Nadimali'],
    'Score (%)': [kasun_score, ranasinghe_score, thakshila_score]
}

df = pd.DataFrame(data)
print(df)
