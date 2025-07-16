import re

def clean_text(text):
    # Remove non-alphanumeric characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def remove_stopwords(text, stopwords):
    words = text.split()
    filtered = [word for word in words if word not in stopwords]
    return ' '.join(filtered)
