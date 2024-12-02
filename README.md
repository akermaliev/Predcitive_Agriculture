# Predictive Analysis for Agriculture Crop Selection

This project leverages machine learning to analyze soil metrics and predict optimal crop types. By combining features such as Nitrogen (N), Phosphorus (P), Potassium (K), and pH, a Logistic Regression model achieves an F1 score of 65% when all features are combined.

## Project Features
- **Classification**: Predict crop types using multi-class Logistic Regression.
- **Feature Evaluation**: Analyze individual soil metrics to identify the most predictive feature (Potassium - K).
- **Visualization**: Generate plots to compare feature performance and model insights.

## Files
- `data/soil_measures.csv`: Dataset used for training and testing.
- `scripts/main.py`: Python code for data preprocessing, feature evaluation, and modeling.
- `results_f1.txt`: Contains the best predictive feature and combined F1 score.

## Requirements
- Python 3.x
- Libraries: pandas, scikit-learn, matplotlib

