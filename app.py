from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your saved model
with open("water_quality.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = [float(x) for x in request.form.values()]

        # Create DataFrame with same column names as training
        input_data = pd.DataFrame([features], columns=[
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ])

        # Predict using the model
        prediction = model.predict(input_data)

        result = "✅ Safe to Drink" if prediction[0] == 1 else "❌ Not Safe"
        return render_template('index.html', prediction_text=f"Water Quality: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
