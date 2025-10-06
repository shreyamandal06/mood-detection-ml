#import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample dataset (text + mood label)
data = [
    ("I am so happy today!", "happy"),
    ("This is the worst day ever", "sad"),
    ("I feel nothing special", "neutral"),
    ("Too much work is stressing me out", "stressed"),
    ("I love my friends", "happy"),
    ("I am really down and depressed", "sad"),
    ("Just another normal day", "neutral"),
    ("My exams are making me so anxious", "stressed"),
    ("Everything is going well", "happy"),
    ("I can't stop crying", "sad"),
    ("I feel relaxed and calm", "happy"),
    ("Life feels meaningless", "sad"),
    ("I’m okay, nothing new", "neutral"),
    ("Deadlines are giving me tension", "stressed"),
    ("I had a wonderful day with family", "happy"),
    ("I don’t want to talk to anyone", "sad"),
    ("Just sitting and chilling", "neutral"),
    ("I’m under a lot of pressure", "stressed"),
    ("I feel so happy today!","happy"),
    ("This is the best day of my life","happy"),
    ("I'm feeling very sad and lonely","sad"),
    ("Nothing seems to cheer me up","sad"),
    ("I'm just sitting quietly","neutral"),
    ("Okay, I understand","neutral"),
    ("I'm under so much pressure","stressed"),
    ("Exams are making me anxious","stressed")

]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

# Save to CSV
df.to_csv("mood_data.csv", index=False)

print("CSV file 'mood_data.csv' created successfully!")

# Load dataset
data = pd.read_csv("mood_data.csv")

# Features and labels
X = data['text']
y = data['label']

# Convert text → numeric features (Bag of Words)
vectorizer = CountVectorizer()
X_features = vectorizer.fit_transform(X)

# Train-test split with stratify to balance classes
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.3, random_state=42, stratify=y
)

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

print("\nNaive Bayes Results:")
print(classification_report(y_test, nb_preds))

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_preds))

# Test with new samples 
test_samples = [
    "I am feeling amazing today!",
    "I have so much stress because of exams",
    "Nothing new, just another day",
    "I am very upset and crying"
]

test_features = vectorizer.transform(test_samples)

print("\nPredictions on new samples:")
for text, label in zip(test_samples, lr_model.predict(test_features)):
    print(f"Text: {text}  --> Predicted mood: {label}")
