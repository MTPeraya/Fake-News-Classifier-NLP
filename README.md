# 📰 Fake News Classifier (NLP and Predictive Analytics)

## Project Overview

This project implements a complete data science pipeline from raw text to a final predictive model to classify news articles as **Real** or **Fake**. Leveraging **Natural Language Processing (NLP)** techniques, the goal is to build a robust classifier that can help detect the spread of misinformation.

The project rigorously follows the analytical structure: **Probability & Statistics $\rightarrow$ Exploratory Data Analysis (EDA) $\rightarrow$ Predictive Analytics.**

## 1\. Data Source

  * **Dataset:** Fake and Real News Dataset $\rightarrow$ [Kaggle Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
  * **Target Variable:** `is_fake` (Binary Classification: 1 for Fake, 0 for Real)
  * **Key Features:** `title`, `text`, `subject`, and `date`.

## 2\. Methodology & Analytical Pipeline

### A. Probability & Statistics (Data Preparation)

This phase focused on cleaning, normalization, and transforming text data into numerical features suitable for machine learning algorithms.

| Task | Outcome |
| :--- | :--- |
| **Data Cleaning** | Handled missing values, removed duplicates, and corrected class imbalances (if necessary). |
| **Text Preprocessing** | Performed **tokenization**, **stop word removal**, and **lemmatization** on the `title` and `text` columns. |
| **Feature Vectorization** | Employed **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform word data into a matrix of numerical importance scores. |

### B. Exploratory Data Analysis (EDA)

The EDA step was used to find distinctive linguistic patterns between the two classes.

  * **Word Frequency Analysis:** Generated **Word Clouds** revealing that fake news tends to use sensational, emotionally charged language, while real news focuses on proper nouns and official titles.
  * **Length Comparison:** Statistical tests showed that, on average, **Fake News articles had longer titles** and a higher incidence of capitalization and punctuation, confirming a sensationalism hypothesis.
  * **Subject Analysis:** Visualized the distribution of the `subject` column to identify which topics are most susceptible to fake news generation.

### C. Predictive Analytics (Modeling & Evaluation)

A range of classification models was tested to determine the most effective strategy for identifying misinformation.

  * **Models Explored:** **Logistic Regression** (as a high-performing baseline for text), **Support Vector Machine (SVM)**, and **Gradient Boosting (XGBoost)**.
  * **Key Metric:** **AUC-ROC (Area Under the Receiver Operating Characteristic curve)** was the primary evaluation metric due to its ability to measure model performance across all classification thresholds, which is crucial for maximizing true positive detection of fake news.
  * **Final Result:** The **[Model Name Here]** achieved an **AUC-ROC score of [X.XX]**, demonstrating strong predictive separation between the classes.

## 3\. Getting Started

To run this analysis locally, you will need **Python 3.x**.

### Dependencies

Install the necessary libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Run the Final Model for a Prediction
The trained model and vectorizer are saved as .joblib files in the models/ directory. You can test the model instantly using the predict.py script.
To run the model, you MUST execute the script from the project root directory:
Navigate to the root directory (the folder containing 'predict.py' and 'src/')
cd path/to/Fake-News-Classifier-NLP/

Run the prediction script
```bash
python predict.py
```

This will run multiple sample headlines and output the model's prediction (Fake or Real) and its certainty score.


### Repository Structure

| Folder/File | Description |
| :--- | :--- |
| `data/` | Stores the raw and cleaned CSV files. **(Ignored from Git)** |
| `notebooks/` | Contains the **Jupyter Notebooks** detailing the analysis steps. |
| `src/` | Holds reusable Python scripts and helper functions (e.g., `text_cleaner.py`). |
| `predict.py` | Standalone Python script to load the model and make real-time predictions on new input text. |
| `README.md` | This document. |

## Findings
- The model performs near-perfectly on the dataset it was trained and tested on.

- However, when evaluated on new, unseen modern headlines, predictions are heavily biased toward the “Fake” label.

This bias is due to:
- Limited vocabulary coverage

- TF-IDF cannot interpret unseen words.

- Dataset bias — fake examples often contain exaggerated, emotional, or outdated language.

### Conclusion
The model is technically robust within its dataset but lacks generalization to modern or out-of-distribution news.

The issue is not a coding error, but a natural limitation of TF-IDF + traditional ML when faced with evolving language.

### Next Steps in the future
- Expand training data with diverse, recent, and balanced real/fake news.

- Increase TF-IDF coverage or switch to a transformer-based model (e.g., BERT) for contextual understanding.

- Re-evaluate model on unseen samples to monitor real-world reliability.