from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def compute_similarity(resume_text, jd_text):
    texts = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]) * 100, 2)

# --- BERT-based similarity ---
def compute_bert_similarity(resume_text, jd_text):
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    model_path = 'models/all-MiniLM-L6-v2'
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
        except Exception:
            pass  # If saving fails, just use the downloaded model
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)
