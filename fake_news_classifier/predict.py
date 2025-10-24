import joblib
import sys
import os
import pandas as pd # Needed for the pd.isna() check inside the cleaner

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.text_cleaner import normalize_text 


MODEL_PATH = 'models/final_xgb_model.joblib'
VECTORIZER_PATH = 'models/tfidf_vectorizer.joblib'


# 1. Load the Model and Vectorizer 
def load_model_artifacts():
    """Loads the saved model and vectorizer."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find model or vectorizer.")
        print("Please ensure both files are saved in the 'models/' directory and paths are correct.")
        sys.exit(1)


# 2. The Main Prediction Function
def predict_news_type(raw_text, model, vectorizer):
    """
    Cleans, vectorizes, and predicts the class of a single piece of text.
    """
    # 1. Cleaning (Must be identical to the training preprocessing)
    cleaned_text = normalize_text(raw_text, lemmatize=True, remove_stopwords=True)
    
    # 2. Vectorization (Using the fitted TF-IDF object)
    # The vectorizer requires input as an iterable (list)
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # 3. Prediction
    # Predict the label (0 or 1)
    prediction = model.predict(text_vectorized)[0]
    
    # Predict the probability of being the positive class (1=Real)
    # The output is [Prob_Fake, Prob_Real]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # 4. Format Output
    if prediction == 1:
        label = "REAL News"
        certainty = probabilities[1]
    else:
        label = "FAKE News"
        certainty = probabilities[0] # Use the probability of the predicted class

    return label, certainty


# --- 3. Example Usage ---
if __name__ == "__main__":
    
    # Load artifacts once
    model, vectorizer = load_model_artifacts()

    # --- Test Cases ---
    test_cases = [
        # Fake / clickbait
        "BREAKING: Celebrity caught in shocking scandal, you won't believe what happens next!",
        "Shocking! Miracle cure for diabetes discovered, doctors hate this secret method!",
        "Aliens spotted on Mars? Scientists confirm extraterrestrial life!",

        # Real / factual
        "The Federal Reserve raised interest rates by 0.25% in its latest policy decision.",
        "NASA successfully launches new telescope to study distant galaxies.",
        "Local community organizes charity event to support homeless families.",

        # Political / controversial
        "Senate votes on new climate change bill amid heated debate.",
        "President signs executive order to improve national cybersecurity.",
        
        # Neutral / tech & science
        "Apple announces iPhone 17 with new AI-powered camera features.",
        "Researchers publish study on the effects of sleep on cognitive function."
    ]


    print("\n=======================================================")
    print("           XGBOOST FAKE NEWS PREDICTOR")
    print("======================================================")
    
    for i, headline in enumerate(test_cases):
        cleaned = normalize_text(headline)
        vec = vectorizer.transform([cleaned])
        pred_label = model.predict(vec)[0]
        pred_proba = model.predict_proba(vec)[0]  # [FAKE, REAL]
        print(f"Headline: {headline}")
        print(f"Prediction: {'FAKE' if pred_label==0 else 'REAL'}")
        print(f"Probabilities: FAKE={pred_proba[0]:.4f}, REAL={pred_proba[1]:.4f}")
        print("--------------------")
        words = headline.split()
vocab = vectorizer.get_feature_names_out()
coverage = sum([w in vocab for w in words]) / len(words)
print(f"Vocabulary coverage: {coverage*100:.2f}%")

