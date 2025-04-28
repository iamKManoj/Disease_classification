from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, String, Column, Integer, MetaData, insert, select, Table, DateTime, quoted_name

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

patient_values = {}

# Load ML model and support files
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("encoding.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

with open("x_filtred_feature_names.pkl", 'rb') as f:
    feature_names = pickle.load(f)

feature_names = feature_names.columns.to_list()

def connect_engine():
	sql_engine = create_engine("mysql+pymysql://root:@localhost/patient_db", echo=True, max_overflow=20)

	metadata = MetaData()
	patient_details = Table(
		'patient_details',
		metadata, 
		Column('id', Integer, autoincrement=True, primary_key=True),
		Column('PatientName', String(500)),
        Column('Gender', String(30)),
		Column('PatientID', Integer),
		Column('Age', Integer),
		Column('BloodGroup', String(500)),
		Column('City', String(500)),
		Column('Contact', Integer()),
		Column('PotentialDisease1', String(500)),
		Column('PotentialDisease2', String(500)),
		Column('UpdatedTime', DateTime),
		*[Column(quoted_name(f"{str(i)}", True), Integer()) for i in feature_names],
	)

	metadata.create_all(sql_engine)
	return sql_engine, patient_details

sql_engine, patient_table = connect_engine()


def insert_values_table(value_dict):
    print("Inserting into DB:", value_dict)
    with sql_engine.connect() as cnn: 
        stmt = insert(patient_table)
        cnn.execute(stmt, value_dict)
        cnn.commit()

# Path to users file
USERS_FILE = "users.json"

# Load users from JSON file
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

# Save users to JSON file
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


# Prediction function
def predict_disease(symptoms):
    input_vector = np.zeros(len(feature_names))
    for symptom in symptoms:
        if symptom in feature_names:
            input_vector[feature_names.index(symptom)] = 1
    if np.sum(input_vector) == 0:
        return "No symptom is entered"

    input_vector = input_vector.reshape(1, -1)
    proba = model.predict_proba(input_vector)[0]
    top2_indices = np.argsort(proba)[-2:][::-1]
    top2_diseases = label_encoder.inverse_transform(top2_indices)
    return top2_diseases[0].capitalize(), top2_diseases[1].capitalize()

# ------------------ ROUTES ------------------

# Login route
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('get_details'))
        else:
            error = "Invalid username or password."
    return render_template('login.html', error=error)

# Sign Up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            error = "Username already exists."
        else:
            users[username] = password
            save_users(users)
            return redirect(url_for('login'))
    return render_template('signup.html', error=error)


#Patient_details page 
@app.route('/details', methods=['GET', 'POST'])
def get_details():
    if request.method=='POST':
        PatientName = request.form['patient_name']
        PatientID = request.form['patient_id']
        Age = request.form['age']
        BloodGroup = request.form['blood_group']
        City = request.form['city']
        Contact = request.form['contact']
        Gender = request.form['gender']
        patient_values["PatientName"] = PatientName 
        patient_values['PatientID'] = PatientID
        patient_values['Age'] = Age
        patient_values['BloodGroup'] = BloodGroup
        patient_values['City'] = City
        patient_values['Contact'] = Contact
        patient_values['Gender'] = Gender
        print(f'--------------------------success--------------\n{patient_values = }')
        return redirect(url_for('index'))
    return render_template('details.html')

# Main prediction page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    prediction = None
    if request.method == 'GET':
        redirect(url_for('get_details'))
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        for feature in feature_names:
            if feature in symptoms: 
                patient_values[feature] = 1
            else: 
                patient_values[feature] = 0

        prediction = predict_disease(symptoms) if symptoms else predict_disease([])
        patient_values["PotentialDisease1"] = prediction[0]
        patient_values["PotentialDisease2"] = prediction[1]
        prediction = f"{prediction[0]}<br>{prediction[1]}"

        patient_values['UpdatedTime'] = datetime.now()
        print(f'---------------{patient_values = }--------')
        # print(f'{prediction = }')
        print('inserting into table....')
        insert_values_table(patient_values)
        print('insert completed...')
    return render_template('index.html', features=feature_names, prediction=prediction)

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
