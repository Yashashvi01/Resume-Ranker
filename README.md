# Resume Ranker App
# [Live Link](https://resume-ranker-3fhfyfjumerpbdetm7q7gu.streamlit.app/)
## 🚀 Project Overview

Welcome to the Resume Ranker App! This project aims to streamline the hiring process by automatically ranking resumes based on their relevance to a given job description. Using advanced Natural Language Processing (NLP) techniques and machine learning, this tool provides an efficient solution for recruiters and HR professionals to quickly identify the best candidates.

## 🔧 Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web framework for building the interactive app
- **NLTK**: Natural Language Processing toolkit for text preprocessing
- **PyPDF2**: Library for reading and extracting text from PDF files
- **Scikit-learn**: Machine learning library for vectorization and similarity calculations
- **WordCloud**: Visualization library for generating word clouds
- **Google Generative AI**: For generating insights from word clouds
- **PIL**: Python Imaging Library for image processing

## 🌟 Features

1. **Resume Upload**: Users can upload multiple PDF resumes.
2. **Job Description Input**: Users can input or paste the job description directly into the app.
3. **Text Preprocessing**: Includes tokenization, stopword removal, and stemming.
4. **Similarity Calculation**: Uses TF-IDF, Cosine Similarity, and Vector Similarity Metric (VSM) to compute how well each resume matches the job description.
5. **Combined Similarity Score**: Aggregates the individual similarity scores into a single, comprehensive score.
6. **Resume Ranking**: Automatically ranks the resumes based on their combined similarity scores.
7. **Word Cloud Generation**: Creates word clouds for both the resumes and job description to visually highlight key terms and skills.
8. **AI-Powered Insights**: Uses Google Generative AI to provide detailed insights on each candidate's strengths and areas for improvement.
9. **Interactive Interface**: Streamlit's interactive features make the app user-friendly and easy to navigate.

## 📈 How It Works

1. **Upload Resumes**: Users can upload multiple PDF resumes at once.
2. **Enter Job Description**: Input the job description in the provided text area.
3. **Process Resumes**: Click on the "Process Resumes" button to analyze and rank the resumes.
4. **View Rankings and Insights**: The app displays the ranked resumes along with detailed insights generated by Google Generative AI.

## 🚀 Future Enhancements

- Adding support for other document formats like DOCX.
- Incorporating additional NLP techniques for better text analysis.
- Enhancing the UI for a more seamless user experience.

## 🛠️ Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Yashashvi01/resume-ranker-app.git
   cd resume-ranker-app
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
3. Run the Streamlit app:
    ```sh
    streamlit run app.py
