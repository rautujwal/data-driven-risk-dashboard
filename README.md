# Data-Driven Risk Dashboard

A dashboard that helps businesses **quantify and manage pricing risks** using predictive analytics, visualizations, and AI-driven recommendations.

## Project Overview

This dashboard uses historical data and machine learning to:

- **Predict pricing outcomes** for products or services using a **Random Forest Regressor**.  
- **Achieved an R² score of 0.95**, demonstrating high predictive accuracy.  
- **Identify high-risk scenarios** that could lead to revenue loss or customer churn.  
- **Simulate “what-if” scenarios** dynamically to assess the impact of different decisions.  
- **Provide AI-driven recommendations** to mitigate risks and optimize outcomes.  
- **Visualize prediction results and risk distributions** with clear charts and plots for better decision-making.  

The project combines a **backend predictive model** (Flask API), a **frontend dashboard interface**, and preprocessed datasets to deliver actionable insights with interactive visualizations.

## Features

- Random Forest-based price prediction  
- Dynamic “what-if” scenario simulation  
- Risk evaluation and scoring  
- AI-based recommendations for business decisions  
- Interactive dashboard interface with **visualizations**  

## Usage

1. Install dependencies:  
\\\ash
pip install -r requirements.txt
\\\

2. Start the backend API:  
\\\ash
python project_api.py
\\\

3. Open \	emplates/project_frontend.html\ in a browser to use the dashboard.

## Notes

- Ensure input data matches the training format for accurate predictions.  
- The model file \pricing_project_model.pkl\ is large; using Git LFS is recommended.
