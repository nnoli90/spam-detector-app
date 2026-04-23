[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🚀 How It Works

1. User enters a text message
2. Message is cleaned (lowercase, remove punctuation/numbers)
3. Text is converted to numerical features using TF-IDF
4. Naive Bayes model predicts SPAM or HAM
5. Result displayed with confidence score

## 📈 Model Training

- **Dataset**: 5,777 SMS messages (4,836 HAM, 941 SPAM)
- **Feature Selection**: 1,000 features (optimized via cross-validation)
- **Train-Test Split**: 80/20 with stratification

## 🔍 Example Predictions

| Message | Prediction | Confidence |
|---------|------------|------------|
| "WINNER! You've won a free iPhone" | SPAM 🚨 | 98.8% |
| "Hey, want to grab coffee tomorrow?" | HAM ✅ | 95.2% |
| "Your package cannot be delivered" | SPAM 🚨 | 76.4% |

## 🏆 Why Naive Bayes?

- Handles imbalanced data well (83.7% HAM, 16.3% SPAM)
- Fast training and prediction
- Excellent for text classification
- Outperformed Logistic Regression in recall (82.98% vs 77.66%)

## 📦 Installation (Local)

```bash
# Clone repository
git clone https://github.com/nnoli90/spam-detector-app.git

# Navigate to project
cd spam-detector-app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py


🚀 Deployment
Deployed on Streamlit Community Cloud - free and easy hosting for Python apps.
https://spam-sms-detector-app-y3ekkjusgxvpwwes9mwjcs.streamlit.app/
