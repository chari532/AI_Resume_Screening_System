# AI Resume Screening System

## Project Overview

An AI-powered resume analysis system developed using Machine Learning and Natural Language Processing (NLP). The application classifies resumes, compares them with job descriptions, and identifies matched and missing technical skills.

## Live Demo

[Launch AI Resume Screening System](https://airesumescreeningsystem-bmug99ddrte7htwiuv4mkm.streamlit.app/)

## Key Features

- Automatic PDF resume text extraction
- NLP-based text preprocessing
- Resume category prediction
- TF-IDF feature extraction
- Cosine Similarity-based resume matching
- Technical skill extraction
- Matched and missing skill identification
- Interactive Streamlit application
- Cloud deployment

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Data visualization |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning |
| PyMuPDF | PDF text extraction |
| Joblib | Model serialization |
| Streamlit | Web application |
| Git & GitHub | Version control |

## Machine Learning Algorithms

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

The best model is selected using stratified cross-validation and macro F1 Score.

## Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Cross-Validation

## Project Workflow

1. Dataset collection
2. Data cleaning
3. Exploratory data analysis
4. NLP text preprocessing
5. TF-IDF vectorization
6. Model training and evaluation
7. PDF text extraction
8. Resume and job description matching
9. Streamlit application development
10. Cloud deployment

## Dataset

[Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

## Installation

Clone the repository:

`git clone https://github.com/chari532/AI_Resume_Screening_System.git`

Navigate to the project directory:

`cd AI_Resume_Screening_System`

Install dependencies:

`pip install -r requirements.txt`

Run the application:

`streamlit run app.py`

## Limitations

- Skill matching depends on predefined technical keywords.
- Scanned PDF documents require OCR support.
- Matching scores measure text relevance, not hiring suitability.

## Future Enhancements

- BERT and semantic similarity
- LLM integration
- Intelligent skill extraction
- OCR support
- Downloadable resume analysis reports
- AI-powered resume improvement suggestions

## Author

**Rachakonda Siva Naga Brahmachari**

AI/ML Engineer | Python | Machine Learning | Data Science

GitHub Profile:-https://github.com/chari532