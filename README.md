# 🧠 AI Resume Matcher with Streamlit UI

An AI-powered application that evaluates resumes against job descriptions and highlights skill gaps using natural language processing. Built with ❤️ using Streamlit for the frontend and powerful transformer models for backend intelligence.

---

## ✅ Key Features

- 📄 Upload resume (PDF format supported)
- 📝 Paste or upload job description
- 🧠 Intelligent match score using **TF-IDF** or **BERT embeddings**
- 📊 Skill gap detection: Highlights missing or weak skills
- 📥 Batch processing support (multiple resumes or JDs)
- 📤 Export analysis results as Excel
- 🖥️ Simple, modern **Streamlit** interface (dark theme)
- 📈 Scalable and customizable for recruiters or job portals

---

## 🚀 Additional Features

- 🔍 Semantic similarity powered by `sentence-transformers`
- 🔧 Modular backend (`resume_parser.py`, `jd_matcher.py`, `utils/`)
- 📂 Organized directory structure for resumes, JDs, and result exports
- 📉 Analytics-ready structure for future employer dashboards
- 🌐 Easily deployable on **Streamlit Cloud**, **Docker**, or **any VM**

---

## 🖥️ Local Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/SriPoojitha31/AI-Resume-Matcher.git
   cd AI-Resume-Matcher
   ```

2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open http://localhost:8501 in your browser**

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your code to GitHub
2. Go to: https://share.streamlit.io
3. Connect your GitHub repository and deploy

**Requirements:**
- Ensure `requirements.txt` is in the root directory
- (Optional) Load BERT model dynamically for zero-setup:

```python
# Add at the top of app.py
from sentence_transformers import SentenceTransformer
import os

if not os.path.exists('models/all-MiniLM-L6-v2'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save('models/all-MiniLM-L6-v2')
```

---

## 🐳 Docker Deployment

1. **Build Docker image**
   ```bash
   docker build -t ai-resume-matcher .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 ai-resume-matcher
   ```

3. **Open http://localhost:8501 in your browser**

---

## 📁 Folder Structure

```
AI-Resume-Matcher/
├── app.py
├── resume_parser.py
├── jd_matcher.py
├── utils/
│   ├── file_utils.py
│   └── text_cleaning.py
├── requirements.txt
├── README.md
├── .gitignore
└── config.yaml       # Add to .gitignore if it contains secrets
```

---

## 📦 Model Download (For Offline Use)

To use BERT-based similarity without internet access:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')
```

Then load it from the local path:

```python
model = SentenceTransformer('models/all-MiniLM-L6-v2')
```

---

## ❌ .gitignore Suggestions

```
venv/
**__pycache__**/
*.pyc
models/
assets/
results/
config.yaml
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🚀 GitHub Project Description
AI Resume Matcher is an intelligent resume screening tool that uses advanced NLP techniques to match candidates with job requirements. Perfect for recruiters, HR professionals, and job seekers looking to optimize their application process.
Key Technologies: Python, Streamlit, BERT, TF-IDF, sentence-transformers, pandas, sklearn

---

## 🌐 Deployed Link 
🚀 Try the live application: https://ai-resume-matcher-kdcecwwjscjbbetnvrmmxs.streamlit.app/
