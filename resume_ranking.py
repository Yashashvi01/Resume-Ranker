import os
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import google.generativeai as genai
from wordcloud import WordCloud
import PIL.Image
import time

# Initialize the Porter stemmer
porter = PorterStemmer()

# Function to preprocess resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [porter.stem(word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Function to read PDF and return text
def read_pdf(file):
    # Create a PyPDF2 reader object
    pdf_reader = PyPDF2.PdfReader(file)

    # Extract text from all pages of PDF
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()

    # Return the text as a string
    return text

# Function to calculate cosine similarity using TF-IDF
def calculate_tfidf_similarity(text1, text2):
    # Preprocess text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Create documents list for vectorization
    documents = [preprocessed_text1, preprocessed_text2]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform documents
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim[0, 1]

# Function to calculate cosine similarity using count vectorizer
def calculate_cosine_similarity(text1, text2):
    # Preprocess text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Create documents list for vectorization
    documents = [preprocessed_text1, preprocessed_text2]

    # Create the Document Term Matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(sparse_matrix, sparse_matrix)

    return cosine_sim[0, 1]

# Function to calculate Vector Similarity Metric (VSM)
def calculate_vsm_similarity(text1, text2):
    # Preprocess text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Convert preprocessed text to vectors
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2]).toarray()

    # Calculate Vector Similarity Metric
    vsm_similarity = np.dot(X[0], X[1]) / (np.linalg.norm(X[0]) * np.linalg.norm(X[1]))

    return vsm_similarity

# Function to calculate the combined similarity score
def calculate_combined_similarity(tfidf_similarity, cos_similarity, vsm_similarity, weights=[0.4, 0.4, 0.2]):
    combined_similarity = (
        tfidf_similarity * weights[0] +
        cos_similarity * weights[1] +
        vsm_similarity * weights[2]
    )
    return combined_similarity * 100

# Define function to select folder and process resumes
def process_resumes(uploaded_files, job_description):

    scores = []

    for file in uploaded_files:
        # Read PDF file
        text1 = read_pdf(file)

        text1 = cleanResume(text1)

        # Calculate similarity scores using different techniques
        tfidf_similarity = calculate_tfidf_similarity(text1, job_description)
        cos_similarity = calculate_cosine_similarity(text1, job_description)
        vsm_similarity = calculate_vsm_similarity(text1, job_description)
        combined_similarity = calculate_combined_similarity(tfidf_similarity,cos_similarity,vsm_similarity)

        # Store scores along with file paths
        scores.append((file, combined_similarity ,text1))

    # Sort resumes based on similarity and sentiment scores
    scores.sort(key=lambda x: (x[1]), reverse=True)

    return scores

def generate_insights(text, job_description, file):
    # Specify a TrueType font path
    font_path = './arial.ttf'

    resume_wordcloud = WordCloud(
        font_path=font_path,
        min_font_size=3, max_words=200, width=800, height=400,
        colormap='viridis', background_color='white'
    ).generate(text)

    jd_wordcloud = WordCloud(
        font_path=font_path,
        min_font_size=3, max_words=200, width=800, height=400,
        colormap='viridis', background_color='white'
    ).generate(job_description)

    # Save the WordCloud images
    resume_wordcloud_filename = f"wordcloud_resume_{os.path.splitext(file.name)[0]}.png"
    resume_wordcloud.to_file(resume_wordcloud_filename)
    jd_wordcloud_filename = "wordcloud_job_description.png"
    jd_wordcloud.to_file(jd_wordcloud_filename)

    # Display WordCloud images using Streamlit
    st.subheader(f"WordCloud for {os.path.splitext(file.name)[0]}")
    st.image(resume_wordcloud_filename)
    st.subheader("WordCloud for Job Description")
    st.image(jd_wordcloud_filename)

    genai.configure(api_key="AIzaSyDc1HZDfUwOShpxdBhgiCM2t1gp9HgbeWE")

    img_res = PIL.Image.open(resume_wordcloud_filename)
    img_jd = PIL.Image.open(jd_wordcloud_filename)
    model = genai.GenerativeModel('gemini-pro-vision')
    
    response = model.generate_content([
        "You are a professional employer evaluating candidates based on a job description using the word cloud of the job description and the resume."
        "Provide concise bullet-point insights on the candidate's qualifications in the following format:\n\n"
        "1. **Candidate's Strengths:**\n"
        "- [List of skills and qualifications matching the job description in not more than 5 points]\n\n"
        "2. **Areas for Improvement:**\n"
        "- [List of missing or weak areas in the candidate's resume in not more than 5 points]\n\n"
        "3. **Overall Score:**\n"
        "- [Score the resume as a strict percentage out of 100 and highlight this score in bold at the end]",
        img_jd,
        img_res,
    ])
    
    response.resolve()
    # Display the insights with animations
    with st.spinner('Generating insights...') :
        time.sleep(2)  # Simulate processing time
        st.subheader(f"Google Gemini Response About {os.path.splitext(file.name)[0]}")
        st.markdown(response.text)

# Main Streamlit app
st.title("Resume Ranker")

# Upload resumes
uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True, type=["pdf"])

# Job description input
job_description = st.text_area("Enter the required job description", key='job_description')

# Number of resumes to display input
num_resumes = st.selectbox("Select number of top resumes to display", [5, 10, 15, 20, 25, 30, 35, 40], key='num_resumes')

if st.button("Process Resumes"):
    if uploaded_files and job_description:
        ranked_resumes = process_resumes(uploaded_files, job_description)
        st.divider()
        for rank, (file, combined_similarity ,resume_text) in enumerate(ranked_resumes[:num_resumes], start=1):
            st.subheader(f"Rank {rank}: {os.path.splitext(file.name)[0]}")
            st.write(f"The Resume is a **{combined_similarity:.2f}%** match with the given job description")
            generate_insights(resume_text, job_description, file)
            st.divider()
        
        # Generate insights for the resumes
    else:
        st.warning("Please enter both folder path and job description")
