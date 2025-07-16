from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)
