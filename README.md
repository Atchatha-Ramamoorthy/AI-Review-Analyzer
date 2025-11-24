
🚀 AI-Powered Product Review Analyzer

An intelligent web application that analyzes product reviews and predicts:

Sentiment → Positive / Negative

Authenticity → Genuine / Fake

Built using NLP + Machine Learning + Flask, it instantly evaluates any product review and gives results with confidence scores.
The system also stores results in a History dashboard that users can revisit.

✨ Features
🧠 Sentiment analysis — Classifies a review as Positive / Negative

🔍 Fake review detection — Detects whether a review is Genuine / Fake

📊 Confidence scores — Displays probability % for both predictions

🕒 Review history — Saves and displays previously analyzed reviews

🎨 Modern UI — Dark-theme dashboard with badges

🚀 Deployment-ready — Works locally and ready for hosting

🛠 Tech Stack
| Component     | Technology                               |
| ------------- | ---------------------------------------- |
| Backend       | Python, Flask                            |
| ML Models     | Logistic Regression / NLP Classification |
| Vectorization | TF-IDF                                   |
| Frontend      | HTML, CSS, Bootstrap                     |
| Storage       | CSV (for history)                        |

📂 Project Structure 
AI_Review_Analyzer/
│
├── Flaskapp/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── data/
│       └── review_history.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── fake_model.pkl
│   └── fake_vectorizer.pkl
│
├── dataset/        # source datasets used for training
├── scripts/        # training notebooks / helper scripts
├── requirements.txt
└── README.md


▶️ Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Start Flask server
cd Flaskapp
python app.py
3️⃣ Open in browser
http://127.0.0.1:5000/

Future Improvements

🔹 Deploy online — Render / Hugging Face / PythonAnywhere / Heroku

🔹 Add multi-language support

🔹 Integrate with Amazon / Flipkart review scrapers

🔹 Export results as Excel / PDF

👩‍💻 Developer
Atchatha Ramamoorthy — MSc Computer Science
📍 United Kingdom
💙 Passionate about NLP and AI-Driven Software
