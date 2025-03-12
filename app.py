# app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from preprocessor import DataPreprocessor  # Import the class

app = Flask(__name__)

# Load the preprocessor and models
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("model_xgb.pkl", "rb") as f:
    model_xgb = pickle.load(f)

with open("model_lgb.pkl", "rb") as f:
    model_lgb = pickle.load(f)

with open("model_cat.pkl", "rb") as f:
    model_cat = pickle.load(f)


@app.route("/", methods=["GET"])
def index():
    return "Ensemble ML API is up and running."


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload containing a list of records.
    For example:
    [
      {
        "Age": 22,
        "Academic Pressure": 7,
        "Work Pressure": 5,
        "CGPA": 8.2,
        "Study Satisfaction": 6,
        "Job Satisfaction": 7,
        "Work/Study Hours": 9,
        "Financial Stress": 3,
        "Gender": "Female",
        "Working Professional or Student": "Student",
        "City": "Mumbai",
        "Family History of Mental Illness": "No",
        "Degree": "Bachelors",
        "Profession": "None",
        "Dietary Habits": "Vegetarian",
        "Have you ever had suicidal thoughts ?": "No",
        "Sleep Duration": "7"
      },
      ...
    ]
    The same preprocessor (and its fitted encoders) is applied before prediction.
    """
    try:
        # Parse JSON payload into a DataFrame
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Transform the input using the preprocessor
        df_transformed = preprocessor.transform(df)
        
        # Get predictions from each model (assumed binary output: 0 or 1)
        preds_xgb = model_xgb.predict(df_transformed).astype(int)
        preds_lgb = model_lgb.predict(df_transformed).astype(int)
        preds_cat = model_cat.predict(df_transformed).astype(int)
        
        # Hard voting ensemble: if at least 2 of 3 models predict 1, then final prediction is 1
        preds_sum = preds_xgb + preds_lgb + preds_cat
        preds_hard_vote = (preds_sum >= 2).astype(int)
        
        # Optionally, include an "id" from input if present
        if "id" in df.columns:
            result = [
                {"id": int(row["id"]), "prediction": int(pred)}
                for row, pred in zip(df.to_dict(orient="records"), preds_hard_vote)
            ]
        else:
            result = preds_hard_vote.tolist()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
