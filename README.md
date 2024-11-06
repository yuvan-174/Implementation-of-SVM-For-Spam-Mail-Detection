# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load data, clean by selecting columns, and preprocess text.
2. Encode labels, vectorize text using TF-IDF.
3. Split data, train SVM classifier.
4. Predict, evaluate accuracy, and generate report.

## Program:
```py
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: YUVAN SUNDAR S
RegisterNumber:212223040250  
*/
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'spam.csv'  # Replace with your actual file path
spam_data = pd.read_csv(file_path, encoding='latin-1')

# Keep only the necessary columns and rename them
spam_data_cleaned = spam_data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})


manual_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
                    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
                    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in manual_stopwords]
    return ' '.join(words)

# Apply preprocessing to messages
spam_data_cleaned['message'] = spam_data_cleaned['message'].apply(preprocess_text)

# Encode labels (spam = 1, ham = 0)
label_encoder = LabelEncoder()
spam_data_cleaned['label'] = label_encoder.fit_transform(spam_data_cleaned['label'])

# Split data into features (X) and labels (y)
X = spam_data_cleaned['message']
y = spam_data_cleaned['label']

# Convert text data to numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Count the number of spam messages in the dataset
spam_count = spam_data_cleaned[spam_data_cleaned['label'] == 1].shape[0]
print(f"Number of spam messages: {spam_count}")

# Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
```

## Output:
![image](https://github.com/user-attachments/assets/72dd77c0-4c0f-4f8e-b494-d0cd83c48123)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
