# AI-Powered Resume Evaluator + Job Matcher

A hackathon-ready MVP that matches resumes to job descriptions using AI, with a fast and clean Streamlit UI.

## Features
- Upload resume PDF
- Paste job description
- Get a match score using cosine similarity or BERT
- Batch processing for resumes and JDs
- Skill gap analysis and suggestions
- Export results to Excel or PDF
- Analytics dashboard for employers
- Modern dark UI

## How to Run (Locally)
1. **Create and activate a virtual environment**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional, for BERT/transformers offline use):**
   - Download the required BERT model (e.g., `all-MiniLM-L6-v2`) using the script below or manually, and place it in the `models/` directory.
   - Update your code to load the model from `models/`.
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
5. Open [http://localhost:8501](http://localhost:8501) in your browser

## Offline Usage
- All dependencies are in `requirements.txt` and can be installed offline if you have the wheels/caches.
- For BERT-based matching, download the model in advance and place it in `models/`.
- No internet is required at runtime if models are present locally.

## Docker Usage
1. **Build the Docker image:**
   ```bash
   docker build -t resume-matcher .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8501:8501 resume-matcher
   ```
3. Open [http://localhost:8501](http://localhost:8501) in your browser

## Model Download (for Offline BERT)
If you use BERT-based matching, download the model before going offline:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')
```
Then, in your code, load it from the local path:
```python
model = SentenceTransformer('models/all-MiniLM-L6-v2')
```

## Folder Structure
```
resume_matcher/
├── app.py
├── resume_parser.py
├── jd_matcher.py
├── utils/
│   ├── file_utils.py
│   └── text_cleaning.py
├── resumes/
├── jds/
├── results/
├── assets/
├── models/           # Place downloaded models here for offline use
├── .gitignore
├── README.md
├── requirements.txt
├── LICENSE
└── config.yaml       # If this contains secrets, add to .gitignore
```

## .gitignore
- The following are already ignored: `venv/`, `__pycache__/`, `*.pyc`, `assets/`, `results/`.
- If `config.yaml` contains secrets, add it to `.gitignore`.

## License
MIT
