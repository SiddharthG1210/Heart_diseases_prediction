🩺 Disease Prediction Web Application

A machine-learning powered web application that predicts multiple diseases based on user inputs and stores prediction history using MongoDB.
The application supports predictions for conditions such as heart disease and diabetes, with a clean web interface built using Flask.

🚀 Features

✅ Disease prediction using trained ML models (.pkl)

✅ Multiple prediction pipelines

✅ User authentication (login & register)

✅ Prediction history stored in MongoDB

✅ Web interface using Flask + HTML/CSS/JS

✅ Ready-to-run with virtual environment support

🧱 Tech Stack

Backend: Python, Flask

Machine Learning: scikit-learn (pickled models)

Database: MongoDB

Frontend: HTML, CSS, JavaScript

Environment Management: Python venv

📁 Project Structure
disease_prediction_app/
│
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── models/                    # Trained ML models (.pkl)
│   ├── *.pkl
│
├── templates/                 # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── *.html
│
├── static/                    # CSS, JS, images
│   ├── style.css
│   ├── script.js
│   └── *.png / *.jfif
│
└── .venv/                     # Virtual environment (ignored by Git)

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/<your-username>/disease_prediction_app.git
cd disease_prediction_app

2️⃣ Create & activate virtual environment
Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate


You should see:

(.venv)

3️⃣ Install dependencies
pip install -r requirements.txt

🗄️ MongoDB Setup
Option A: Local MongoDB

Install MongoDB Community Server

Make sure MongoDB is running on:

mongodb://localhost:27017

Option B: MongoDB Atlas (Cloud)

Create a free cluster at https://www.mongodb.com/atlas

Get your connection URI

Update your MongoDB connection string in app.py

Example:

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["disease_prediction_db"]


⚠️ Do NOT commit MongoDB credentials.
Use environment variables if deploying.

▶️ Run the Application
python app.py


Then open your browser and visit:

http://127.0.0.1:5000

📊 Models

Trained ML models are stored in the models/ directory

Models are loaded using pickle

These are required for predictions to work

If you retrain models, replace the .pkl files in models/.

🔐 Environment Variables (Recommended)

Create a .env file (not committed):

MONGO_URI=mongodb://localhost:27017
SECRET_KEY=your_secret_key


And load it in app.py.

❌ What is NOT committed (by design)

Virtual environments (.venv/, venv/)

Python cache files

Local secrets

These are ignored via .gitignore.

🧠 Notes for Developers

Use .venv for all development

Update requirements.txt after installing new packages:

pip freeze > requirements.txt


Commit changes incrementally (avoid git add . blindly)

📌 Future Improvements (Optional)

Docker support

Model versioning

API endpoints

Better validation & error handling

Role-based access

📜 License

This project is for educational and demonstration purposes.
