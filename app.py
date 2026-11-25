import streamlit as st
import joblib
import fitz
import re
import nltk

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_models():
    model = joblib.load('resume_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


skills = [
    'python', 'sql', 'pandas', 'numpy',
    'machine learning', 'deep learning',
    'power bi', 'excel', 'statistics',
    'matplotlib', 'seaborn', 'scikit-learn',
    'tensorflow', 'pytorch', 'tableau',
    'data analysis', 'nlp', 'git',
    'streamlit', 'aws', 'spark', 'pyspark',
    'langchain', 'rag', 'llm', 'docker',
    'fastapi', 'flask', 'mongodb'
]


def clean_resume(text):

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    return text


def extract_pdf_text(uploaded_file):

    text = ""

    with fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    ) as document:

        for page in document:
            text += page.get_text() + "\n"

    return text



def extract_skills(text):

    text = text.lower()
    extracted_skills = set()

    for skill in skills:

        pattern = (
            r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
        )

        if re.search(pattern, text):
            extracted_skills.add(skill)

    return extracted_skills



st.title("📄 AI Resume Screening System")

st.write(
    "Analyze your resume and compare it "
    "with a job description using AI."
)

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Enter Job Description",
    height=200
)

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            resume_text = extract_pdf_text(uploaded_file)

        except Exception:

            st.error("Unable to read this PDF.")
            st.stop()

        if not resume_text.strip():

            st.error("No readable text found in the PDF.")
            st.stop()

        cleaned_resume = clean_resume(resume_text)
        cleaned_jd = clean_resume(job_description)

        
        resume_vector = tfidf.transform([cleaned_resume])
        jd_vector = tfidf.transform([cleaned_jd])

        if resume_vector.nnz == 0 or jd_vector.nnz == 0:

            st.warning("Insufficient recognized text for analysis.")
            st.stop()

     
        category = model.predict(resume_vector)[0]

      
        similarity = cosine_similarity(
            resume_vector, jd_vector
        )[0][0] * 100

    
        resume_skills = extract_skills(resume_text)
        required_skills = extract_skills(job_description)

        matched = resume_skills & required_skills
        missing = required_skills - resume_skills

        skill_score = (
            len(matched) / len(required_skills) * 100
            if required_skills else None
        )

      
        st.subheader("Resume Analysis Report")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Category",
            category
        )

        col2.metric(
            "Text Matching Score",
            f"{similarity:.2f}%"
        )

        col3.metric(
            "Skill Matching Score",
            f"{skill_score:.2f}%"
            if skill_score is not None else "N/A"
        )

        if skill_score is not None:
            st.progress(float(skill_score / 100))

        left, right = st.columns(2)

        with left:

            st.subheader("Matched Skills")

            for skill in sorted(matched):
                st.success(skill)

            if not matched:
                st.info("No recognized matching skills.")

        with right:

            st.subheader("Missing Skills")

            for skill in sorted(missing):
                st.warning(skill)

            if not missing:
                st.info("No unmatched recognized skills.")

        with st.expander("View Extracted Resume Text"):
            st.text(resume_text)

        st.caption(
            "Scores measure text relevance and recognized "
            "skill coverage, not hiring suitability."
        )