from flask import Flask, render_template,request,jsonify,session
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS
import sounddevice as sd
import os
import soundfile as sf
import speech_recognition as sr
import random
from difflib import SequenceMatcher  

app = Flask(__name__)
app.secret_key = '5500'

@app.route("/")
def index():
    session['stage'] = 'hastalik_tahmini'
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def getUserResponse():
    if request.method == "POST":
        user_message = request.form["userMessage"]
        if session['stage'] == 'hastalik_tahmini':
            response = hastalik(user_message)
            session['hastalik'] = response  
            session['stage'] = 'chatbot'
            if response=="no_valid":
                session['stage'] = 'hastalik_tahmini'
                responseE = no_valid(user_message)
                return "Warning: {} is not recognized as a symptom. Try again.".format(responseE)
            return "Predicted disease: {}".format(response)

        elif session['stage'] == 'chatbot':
            response_a = chatbot(session['hastalik'], user_message)
            return "{}".format(response_a)
        else:
            return "Error: Unknown stage."
    else:
        return "Error: Only POST requests are allowed."

@app.route("/reset", methods=["POST"])
def reset():
    session['stage'] = 'hastalik_tahmini'
    return "Session reset"

@app.route("/finish", methods=["POST"])
def finish():
    return jsonify({"response": "Sohbet sona erdi."}), 200, {'Refresh': '5; url=/'}

dataset = pd.read_json("C:\\Users\\furka\\Desktop\\boot\\test_data.json")

model = load_model("hastalik_tahmin_modeli.h5")

label_encoder = LabelEncoder()
label_encoder.fit(dataset["prognosis"])

all_symptoms = list(dataset.columns[:-2])

def similar_words(user_input, known_symptom):
    similarity = SequenceMatcher(None, user_input, known_symptom).ratio()
    return similarity >= 0.7

def similar_ques(user_input, questions):
    similarity = SequenceMatcher(None, user_input, questions).ratio()
    return similarity >= 0.7 

responses_dataset = pd.read_json("C:\\Users\\furka\\Desktop\\boot\\json16k_cevap_updated.json")

def hastalik(hst):
    user_input_str = str(hst)
    user_input_str = user_input_str.replace('%20', ' ')
    user_input_list = [symptom.strip() for symptom in user_input_str.split('%2C')]
    symptom_dict = {symptom: 0 for symptom in all_symptoms}
    valid_symptoms = False
    while True:
        for user_symptom in user_input_list:
            if user_symptom:  
                matched = False
                for known_symptom in all_symptoms:
                    if similar_words(user_symptom, known_symptom):
                        symptom_dict[known_symptom] = 1
                        valid_symptoms = True
                        matched = True
                        break
                if not matched:   
                    return("no_valid")

        if valid_symptoms:
            user_input_array = list(symptom_dict.values())
            user_input_df = pd.DataFrame([user_input_array])
            user_input_array = np.expand_dims(user_input_df.values, axis=2).astype('float32')
            prediction = model.predict(user_input_array)
            predicted_label_index = np.argmax(prediction)
            predicted_label = label_encoder.classes_[predicted_label_index]
            return predicted_label
        else:
            return ("No valid symptoms entered. Please try again.")

def no_valid(hsst):
    user_input_str = str(hsst)
    user_input_str = user_input_str.replace('%20', ' ')
    user_input_list = [symptom.strip() for symptom in user_input_str.split('%2C')]
    symptom_dict = {symptom: 0 for symptom in all_symptoms}
    while True:
        for user_symptom in user_input_list:
            if user_symptom:  
                matched = False
                for known_symptom in all_symptoms:
                    if similar_words(user_symptom, known_symptom):
                        symptom_dict[known_symptom] = 1                     
                        matched = True
                        break
                if not matched:   
                    return(user_symptom)

def chatbot(tag,msg):
    message = msg.replace('%20', ' ')
    if not isinstance(message, str): 
        return("An error occurred while predicting the disease.")    
    if responses_dataset[responses_dataset['tag'] == tag].empty:
        return("I couldn't find any information related to the predicted disease.")       
    responses = responses_dataset[responses_dataset['tag'] == tag]['Questions'].values[0]     
    best_matches = [(similar_ques(message, response['Question']), response) for response in responses]
    best_matches.sort(key=lambda x: x[0], reverse=True)  
    for score, response in best_matches:
        if score > 0.5:
            return (random.choice(response['Answers']))  
        matched = False
        for response in responses:
            if similar_ques(message, response['Question']):
                matched = True
                return (random.choice(response['Answers']))              
        if matched==False:
            return ("I am sorry, I could not find any information related to your question.")

if __name__=='__main__':
    app.run()