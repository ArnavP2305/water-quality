from flask import Flask, render_template, request
import pickle
import numpy as np

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
        features_array = np.array([features])
        
        # Prediction
        prediction = model.predict(features_array)[0]

        # Get probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_array)[0]
            safe_prob = round(proba[1] * 100, 2)   # probability of safe (class 1)
            unsafe_prob = round(proba[0] * 100, 2) # probability of not safe (class 0)
            result = f"Water Quality: {'✅ Safe' if prediction == 1 else '❌ Not Safe'} " \
                     f"(Confidence → Safe: {safe_prob}%, Not Safe: {unsafe_prob}%)"
        else:
            result = f"Water Quality: {'✅ Safe' if prediction == 1 else '❌ Not Safe'}"
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
