from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from datetime import datetime
from flask_login import current_user

app = Flask(__name__)

# Stuff for login session

app.config["MONGO_URI"] = "mongodb://localhost:27017/disease_prediction_app"
app.secret_key = "your_secret_key"  # Change this to a secure key
mongo = PyMongo(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = str(user_id)  # Ensure user_id is a string
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})  # Convert back to ObjectId
    if user:
        return User(user_id=str(user["_id"]), username=user["username"])
    return None

# End of login session stuff

# Load the model and scaler from the 'models' folder
model_path = os.path.join('models', 'logistic_regression_model (4).pkl')
scaler_path = os.path.join('models', 'MinMaxScaler_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/heart-disease-test')
def heart_disease_test():
    return render_template('heart_disease.html')  # ✅ Correct template name



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        
        # Create feature array for prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                              thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Determine result message
        if prediction[0] == 1:
            result = "Positive"
            message = "The model predicts that the patient may have heart disease."
        else:
            result = "Negative"
            message = "The model predicts that the patient does not have heart disease."

        
        if current_user.is_authenticated:
            input_data = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
                'ca': ca, 'thal': thal
            }
            entry = {
                'input_data': input_data,
                'result': result,
                'probability': round(probability * 100, 2),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            mongo.db.users.update_one(
                {'_id': ObjectId(current_user.id)},
                {'$push': {'test_history.heart_disease': entry}}
            )

        return render_template('heart_disease_result.html', 
                              prediction=result,
                              probability=round(probability * 100, 2),
                              message=message) 
    

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease_api():
    """API endpoint for prediction"""
    data = request.json
    
    # Extract features
    features = np.array([[
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'],
        data['thal']
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    return jsonify({
        'prediction': int(prediction),
        'probability': round(probability * 100, 2),
        'has_heart_disease': bool(prediction == 1)
    })
  
@app.route('/heart-disease-history')
@login_required
def heart_disease_history():
    user = mongo.db.users.find_one({'_id': ObjectId(current_user.id)})
    history = user.get('test_history', {}).get('heart_disease', [])
    sorted_history = sorted(history, key=lambda x: x['date'], reverse=True)
    return render_template('heart_disease_history.html', history=sorted_history)
    

# Load trained model & scaler
model2 = joblib.load("diabetes_model_fixed (2).pkl")
scaler2 = joblib.load("scaler.pkl")

@app.route('/diabetes-test')
def diabetes_test():
    return render_template('diabetes.html')

@app.route('/diabetes-predict', methods=['POST'])
def diabetes_predict():
    if request.method == 'POST':
        # Get form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        # Create feature array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, diabetes_pedigree, age]])

        # Scale features
        scaled_features = scaler2.transform(features)

        # Make prediction
        prediction = model2.predict(scaled_features)
        probability = model2.predict_proba(scaled_features)[0][1]

        # Determine result message
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        message = "The model predicts that the patient has diabetes." if prediction[0] == 1 else "The model predicts that the patient does not have diabetes."

        # ✅ Save history in MongoDB
        if current_user.is_authenticated:
            username = current_user.username
            input_data = {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'bmi': bmi,
                'diabetes_pedigree': diabetes_pedigree,
                'age': age
            }
            entry = {
                'input_data': input_data,
                'result': result,
                'probability': round(probability * 100, 2),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            # Update user's test history
            mongo.db.users.update_one(
                {'username': username},
                {'$push': {'test_history.diabetes': entry}}
            )

        return render_template('diabetes_result.html', 
                               prediction=result,
                               probability=round(probability * 100, 2),
                               message=message)
# route for diabetes history 
@app.route('/diabetes-history')
@login_required
def diabetes_history():
    
        user = mongo.db.users.find_one({'_id': ObjectId(current_user.id)})
        diabetes_history = user.get('test_history', {}).get('diabetes', [])
        # Sort by date, most recent first
        sorted_history = sorted(diabetes_history, key=lambda x: x['date'], reverse=True)
        return render_template('diabetes_history.html', history=sorted_history)
    



# Load the trained model and scaler
model3 = joblib.load('claude_heart_disease_model (1).pkl')
scaler3 = joblib.load('claude_scaler (1).pkl')


@app.route('/heart-attack-test')
def heart_attack_test():
    return render_template('heart_attack.html')

@app.route('/predict_heart_attack', methods=['POST'])
def predict_heart_attack():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        thalach = float(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        exang = int(request.form['exang'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Binary feature engineering (same as training)
        high_chol = 1 if chol > 200 else 0
        high_trestbps = 1 if trestbps > 140 else 0

        # Prepare features (excluding 'age_group' as done in training)
        features = np.array([[age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, 
                              restecg, exang, slope, ca, thal, high_chol, high_trestbps]])

        # Scale numerical features
        features[:, :5] = scaler3.transform(features[:, :5])

        # Make prediction
        prediction = model3.predict(features)
        probability = model3.predict_proba(features)[0][1]

        # Determine result message
        if prediction[0] == 1:
            result = "High Risk"
            message = "The model predicts a high risk of heart attack."
        else:
            result = "Low Risk"
            message = "The model predicts a low risk of heart attack."

        if current_user.is_authenticated:
            input_data = {
                'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
                'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg, 'exang': exang,
                'slope': slope, 'ca': ca, 'thal': thal,
                'high_chol': high_chol, 'high_trestbps': high_trestbps
            }
            entry = {
                'input_data': input_data,
                'result': result,
                'probability': round(probability * 100, 2),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            mongo.db.users.update_one(
                {'_id': ObjectId(current_user.id)},
                {'$push': {'test_history.heart_attack': entry}}
            )

        return render_template('heart_attack_result.html', 
                              prediction=result,
                              probability=round(probability * 100, 2),
                              message=message) 

@app.route('/heart-attack-history')
@login_required
def heart_attack_history():
    user = mongo.db.users.find_one({'_id': ObjectId(current_user.id)})
    history = user.get('test_history', {}).get('heart_attack', [])
    sorted_history = sorted(history, key=lambda x: x['date'], reverse=True)
    return render_template('heart_attack_history.html', history=sorted_history)



# routes for login session
from bson.objectid import ObjectId

# Register route with additional fields
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        dob = request.form["dob"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Check if passwords match
        if password != confirm_password:
            return "Passwords do not match!"

        # Check if username or email already exists
        existing_user = mongo.db.users.find_one({"$or": [{"username": username}, {"email": email}]})
        if existing_user:
            if existing_user.get("username") == username:
                return "Username already exists!"
            if existing_user.get("email") == email:
                return "Email already exists!"

        # Hash password and save user data
        hashed_password = generate_password_hash(password)
        user_data = {
            "username": username,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "dob": dob,
            "password": hashed_password
        }
        user_id = mongo.db.users.insert_one(user_data).inserted_id
        user = User(user_id=str(user_id), username=username)
        login_user(user)

        return redirect(url_for("test_selection"))  # Redirect after registration

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = mongo.db.users.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            login_user(User(user_id=str(user["_id"]), username=username))
            return redirect(url_for("test_selection"))  # Redirect after login
        
        return "Invalid username or password!"

    return render_template("login.html")


@app.route('/test-selection')
@login_required
def test_selection():
    return render_template('test_selection.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))



if __name__ == '__main__':
    app.run(debug=True)
