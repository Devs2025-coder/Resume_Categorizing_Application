# 📄 AI Resume Categorization Application

This is a Streamlit-based web application that automatically extracts and categorizes resumes using NLP and machine learning models. It reads uploaded resumes in PDF format, parses their content, and classifies them into categories such as **Data Science**, **Web Development**, etc.

---

## 🚀 Features

- 🧠 **Resume Categorization** using a trained ML model
- 📄 **PDF Resume Parsing** with `pypdf`
- 📊 **Data Handling** using `pandas`
- ⏰ **Timestamp logging** for actions using `datetime`
- 🖥️ **Web Interface** built with `Streamlit`
- 💅 **Custom CSS Styling** for a better UI

---

## 🛠️ Tech Stack

| Component       | Description                          |
|----------------|--------------------------------------|
| Python          | Core language                        |
| Streamlit       | Web frontend framework               |
| pypdf           | PDF resume text extraction           |
| pandas          | Resume data handling and processing  |
| scikit-learn    | ML model training and inference      |
| pickle          | Model serialization (standard lib)   |

---

## 📁 Project Structure
Resume Categorization Application/
│
├── app.py # Main Streamlit app
├── requirements.txt # Required Python packages
├── model.pkl # Trained ML model
├── tfidf.pkl # TF-IDF vectorizer
├── Resume.csv # Dataset (optional)
├── categorized_resumes/ # Folder to save categorized files
├── Resumes/ # Folder to upload raw resumes
└── README.md # You're reading it!

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/resume-categorizer.git
   cd resume-categorizer

## my web-app link
https://resumecategorizingapplication-8tpuydhpqdh8wmbwelmc9m.streamlit.app/
