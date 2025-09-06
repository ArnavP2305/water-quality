from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your saved model
with open("water_quality.pkl", "rb") as f:
    model = pickle.load(f)

# Define the same feature names you used during training
feature_names = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        values = list(request.form.values())

        # Check for empty inputs
        if "" in values:
            return render_template(
                'index.html',
                prediction_text="⚠️ Please fill all fields before submitting."
            )

        # Convert values to float
        features = [float(x) for x in values]

        # Convert to DataFrame with column names
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Prediction
        prediction = model.predict(features_df)[0]

        # Get probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]
            safe_prob = round(proba[1] * 100, 2)
            unsafe_prob = round(proba[0] * 100, 2)
            result = f"Water Quality: {'✅ Safe' if prediction == 1 else '❌ Not Safe'} " \
                     f"(Confidence → Safe: {safe_prob}%, Not Safe: {unsafe_prob}%)"
        else:
            result = f"Water Quality: {'✅ Safe' if prediction == 1 else '❌ Not Safe'}"
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
