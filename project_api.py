from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import numpy as np
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

# Model and API setup
api_key = "hidden_for_privacy"
model = "llama-3.3-70b-versatile"
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}

final_model_pipeline = load("pricing_project_model.pkl")
one_hot_columns = load("one_hot_columns.pkl")
ordinal_columns = load("ordinal_columns.pkl")
scale_columns = load("scale_columns.pkl")

classify_risk = lambda z_score: (
    "Very High Risk (Customer Churn Risk)" if z_score > 1 else
    "Very Low Price (Undervaluation Risk)" if z_score < -1 else
    "Moderate Price (Low Risk)"
)

@app.route('/')
def index():
    return render_template("project_frontend.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json
        features = user_input.get("features")
        adjustment = user_input.get("adjustment", 0)
        finalize = user_input.get("finalize", False)

        df = pd.DataFrame([features])
        df.columns = df.columns.str.lower().str.strip()

        base_price = final_model_pipeline.predict(df)[0]
        rf_model = final_model_pipeline.named_steps['model']
        transformer = final_model_pipeline.named_steps['transformer']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-5:]
        feature_names = transformer.get_feature_names_out()
        X_transformed = transformer.transform(df)

        tree_predictions = np.array([tree.predict(X_transformed) for tree in rf_model.estimators_])
        pred_std = tree_predictions.std()
        pred_mean = tree_predictions.mean()
        price = base_price + (adjustment / 100) * base_price
        z_score = (price - pred_mean) / pred_std
        risk = classify_risk(z_score)

        # Visualization
        fig, axis = plt.subplots(ncols=2, figsize=(12, 8))
        sns.histplot(x=tree_predictions.flatten(), bins=10, alpha=1, color='black', kde=True, ax=axis[0])
        axis[0].axvline(price, color='red', linestyle='--', lw=2, label=f'Predicted Price: {price:.2f}')
        axis[0].set_title('Model Prediction at a Glance', fontsize=14, fontweight='bold')
        axis[0].set_xlabel('Price (INR)')
        axis[0].set_ylabel('Frequency')
        sns.heatmap(pd.DataFrame([importances[indices]], columns=feature_names[indices], index=['importance']).transpose(),
                    lw=1, ax=axis[1])
        axis[1].set_title('Most Important Features of your Car', fontsize=14, fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # AI Report
        report = ""
        if finalize:
            prompt = f"""
            The forecasting model predicted the price of a {features.get('car_name')} as INR {price:.2f}.
            The Z_score value is {z_score:.2f}.
            The risk classification is: {risk}.
            The input features are: {features}.
            The important features are: {feature_names[indices]}
            Provide a clear, concise explanation (~180 words) of why this price might be a great deal/ok deal or risky.
            Mention potential customer churn or revenue loss if risky or potential growth if not risky, and highlight key features.
            If the risk is moderate, suggest two recommendations in bullet points.
            If no risk, explain why this is a great deal.
            Compare with at least one other dealer in India if possible and give price range.
            """
            data_api = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post(url, headers=headers, json=data_api)
            try:
                report = response.json()["choices"][0]["message"]["content"]
            except KeyError:
                report = f"[AI API Error] {response.text}"

        return jsonify({
         "base_price": base_price,
         "car_name": features.get("car_name"),
         "price": price,
         "risk": risk,
         "visualization": img_base64,
         "report": report,
         "finalized": finalize
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
