from flask import Flask, render_template, request
from PIL import Image
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
with open('toxic_vect.pkl', 'rb') as f:
    toxic_vect = pickle.load(f)
with open('toxic_model.pkl', 'rb') as f:
    toxic_model = pickle.load(f)
with open('severe_toxic_vect.pkl', 'rb') as f:
    severe_toxic_vect = pickle.load(f)
with open('severe_toxic_model.pkl', 'rb') as f:
    severe_toxic_model = pickle.load(f)
with open('threat_vect.pkl', 'rb') as f:
    threat_vect = pickle.load(f)
with open('threat_model.pkl', 'rb') as f:
    threat_model = pickle.load(f)
with open('obscene_vect.pkl', 'rb') as f:
    obscene_vect = pickle.load(f)
with open('obscene_model.pkl', 'rb') as f:
    obscene_model = pickle.load(f)
with open('insult_vect.pkl', 'rb') as f:
    insult_vect = pickle.load(f)
with open('insult_model.pkl', 'rb') as f:
    insult_model = pickle.load(f)
with open('identity_hate_vect.pkl', 'rb') as f:
    identity_hate_vect = pickle.load(f)
with open('identity_hate_model.pkl', 'rb') as f:
    identity_hate_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']

    def format_probability(prob):
        if prob <= 0.35:
            return 0
        else:
            return max(0, prob)

    # Define result dictionary
    result = {}

    # Toxic
    toxic_vect_text = toxic_vect.transform([comment])
    toxic_prob = round(toxic_model.predict_proba(toxic_vect_text)[:, 1][0], 2)
    toxic_prediction = toxic_model.predict(toxic_vect_text)
    toxic_prob = format_probability(toxic_prob)
    result['Toxic'] = f"{toxic_prob} {'True' if toxic_prob >= 0.65 else ('Mid' if 0.35 < toxic_prob < 0.65 else 'False')}"

    # Severe toxic
    severe_toxic_vect_text = severe_toxic_vect.transform([comment])
    severe_toxic_prob = round(severe_toxic_model.predict_proba(severe_toxic_vect_text)[:, 1][0], 2)
    severe_toxic_prediction = severe_toxic_model.predict(severe_toxic_vect_text)
    severe_toxic_prob = format_probability(severe_toxic_prob)
    result['Severe Toxic'] = f"{severe_toxic_prob} {'True' if severe_toxic_prob >= 0.65 else ('Mid' if 0.35 < severe_toxic_prob < 0.65 else 'False')}"

    # Threat
    threat_vect_text = threat_vect.transform([comment])
    threat_prob = round(threat_model.predict_proba(threat_vect_text)[:, 1][0], 2)
    threat_prediction = threat_model.predict(threat_vect_text)
    threat_prob = format_probability(threat_prob)
    result['Threat'] = f"{threat_prob} {'True' if threat_prob >= 0.65 else ('Mid' if 0.35 < threat_prob < 0.65 else 'False')}"

    # Obscene
    obscene_vect_text = obscene_vect.transform([comment])
    obscene_prob = round(obscene_model.predict_proba(obscene_vect_text)[:, 1][0], 2)
    obscene_prediction = obscene_model.predict(obscene_vect_text)
    obscene_prob = format_probability(obscene_prob)
    result['Obscene'] = f"{obscene_prob} {'True' if obscene_prob >= 0.65 else ('Mid' if 0.35 < obscene_prob < 0.65 else 'False')}"

    # Insult
    insult_vect_text = insult_vect.transform([comment])
    insult_prob = round(insult_model.predict_proba(insult_vect_text)[:, 1][0], 2)
    insult_prediction = insult_model.predict(insult_vect_text)
    insult_prob = format_probability(insult_prob)
    result['Insult'] = f"{insult_prob} {'True' if insult_prob >= 0.65 else ('Mid' if 0.35 < insult_prob < 0.65 else 'False')}"

    # Identity hate
    identity_hate_vect_text = identity_hate_vect.transform([comment])
    identity_hate_prob = round(identity_hate_model.predict_proba(identity_hate_vect_text)[:, 1][0], 2)
    identity_hate_prediction = identity_hate_model.predict(identity_hate_vect_text)
    identity_hate_prob = format_probability(identity_hate_prob)
    result['Identity Hate'] = f"{identity_hate_prob} {'True' if identity_hate_prob >= 0.65 else ('Mid' if 0.35 < identity_hate_prob < 0.65 else 'False')}"

    
    
    
    # Calculate the counts of True, False, and Mid for each category
    counts = {'True': 0, 'False': 0, 'Mid': 0}

    # Update counts for each category
    for key, value in result.items():
        if value.endswith('True'):
            counts['True'] += 1
        elif value.endswith('False'):
            counts['False'] += 1
        elif value.endswith('Mid'):
            counts['Mid'] += 1

    # Determine the final conclusion based on the counts
    if counts['True'] == 6:
        final_conclusion = 'Chắc chắn độc hại'
    elif counts['True'] >= 1 and counts['True'] <= 2 and counts['False'] > 0:
        final_conclusion = 'Có khả năng độc hại'
    elif counts['True'] >= 1 and counts['True'] <= 2 and counts['False'] == 0:
        final_conclusion = 'Có khả năng độc hại'
    elif counts['False'] == 6:
        final_conclusion = 'Không độc hại'
    elif counts['True'] == 0 and counts['Mid'] > counts['False']:
        final_conclusion = 'Có khả năng độc hại'
    elif counts['True'] == 0 and counts['False'] > counts['Mid']:
        final_conclusion = 'Không độc hại'
    else:
        final_conclusion = 'Có khả năng độc hại nhưng thấp'

    # Add final conclusion to the result dictionary
    result['Final Conclusion'] = final_conclusion
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()