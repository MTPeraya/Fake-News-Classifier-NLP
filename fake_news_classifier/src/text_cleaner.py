import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Use global variables, but initialize them lazily
lemmatizer = None
stop_words = None

def initialize_nltk_resources():
    """Initializes NLTK resources (Lemmatizer, Stopwords) after checks."""
    global lemmatizer, stop_words
    
    # Only initialize if it hasn't been done yet
    if lemmatizer is None:
        try:
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            print("NLTK resources successfully initialized.")
        except LookupError:
            # Re-raise error with instructions if download was missed
            raise LookupError(
                "NLTK resources (stopwords/wordnet/punkt) not found. "
                "Please run nltk.download('...') in your notebook first."
            )
            
def normalize_text(text, lemmatize=True, remove_stopwords=True):
    """
    Performs comprehensive text cleaning and normalization.
    """
    # Ensure resources are ready before proceeding
    if lemmatizer is None:
        initialize_nltk_resources() 

    if pd.isna(text):
        return ""

    # 1. Case Conversion (Lowercase)
    text = text.lower()
    
    # ... (rest of your normalize_text function remains the same) ...
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) 
    words = word_tokenize(text)

    # 5. Stopword Removal and Lemmatization
    cleaned_words = []
    for word in words:
        if remove_stopwords and word in stop_words:
            continue
        
        if len(word) <= 1:
            continue
            
        if lemmatize:
            word = lemmatizer.lemmatize(word)
        
        cleaned_words.append(word)
    
    return ' '.join(cleaned_words)
