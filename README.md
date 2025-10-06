# Mood Detection ML Project

A simple machine learning project to classify human moods (happy, sad, neutral, stressed) from short text inputs. This uses basic NLP techniques to analyze sentiment in text data.

## Objective
The goal is to build a text classifier that predicts a person's mood based on their written statements. For example, "I am so happy today!" should be classified as "happy". This project demonstrates entry-level ML for sentiment analysis, which can be extended to mental health apps or chatbots. I chose this to explore text processing and model evaluation.

## Tech Stack
- **Language**: Python 3.13
- **Libraries**:
  - `pandas`: For data handling and CSV operations.
  - `scikit-learn`: For text vectorization (CountVectorizer), model training (Naive Bayes, Logistic Regression), and evaluation (train-test split, classification report).

## Implementation Details
1. **Data Preparation**:
   - Created a sample dataset of text-mood pairs (e.g., happy/sad statements).
   - Saved as `mood_data.csv` for reproducibility.
   - Features (X): Text inputs. Labels (y): Mood categories (happy, sad, neutral, stressed).

2. **Text Processing**:
   - Used `CountVectorizer` to convert text into a Bag-of-Words matrix (numeric features based on word counts).

3. **Model Training and Evaluation**:
   - Split data: 70% train, 30% test (stratified for class balance).
   - Trained two models:
     - **Naive Bayes (MultinomialNB)**: Simple probabilistic classifier, good for text data.
     - **Logistic Regression**: Linear model for multi-class classification.
   - Evaluated using `classification_report` (precision, recall, F1-score per class).

4. **Predictions**:
   - Tested on new unseen samples.
   - Logistic Regression performed better in my runs (higher accuracy).

5. **How to Run**:
```bash
pip install pandas scikit-learn
```
   - Run: `python mood_detection.py`
   - Output: Classification reports and predictions in the console.

## Future Improvements
- Add more data for better accuracy.
- Use advanced NLP (e.g., TF-IDF or BERT).
- Build a web app with Streamlit for interactive predictions.

