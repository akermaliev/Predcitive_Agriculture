# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
crops = pd.read_csv("data/soil_measures.csv")
print("Dataset loaded successfully!")
print(crops.head())  # Display the first few rows of the dataset

# Separate features and target variable
X = crops[["N", "P", "K", "ph"]]  # Features
y = crops["crop"]  # Target

# Check for missing values
print("Missing values in the dataset:", crops.isna().sum())

# Check unique target classes (crops)
print("Unique crop types:", crops["crop"].unique())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Initialize variables for tracking performance
feature_performance = {}  # Store F1 scores for each feature
highest_f1 = 0
best_feature_name = ""

# Evaluate each feature individually
features = ["N", "P", "K", "ph"]
f1_scores = []  # To store F1 scores for visualization

for feature in features:
    print(f"Evaluating feature: {feature}")
    X_train_feature = X_train[[feature]]  # Select a single feature
    X_test_feature = X_test[[feature]]

    # Scale the feature (optional for better performance)
    scaler = StandardScaler()
    X_train_feature = scaler.fit_transform(X_train_feature)
    X_test_feature = scaler.transform(X_test_feature)

    # Train logistic regression model
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_train_feature, y_train)

    # Predict and calculate F1 score
    y_pred = model.predict(X_test_feature)
    f1 = f1_score(y_test, y_pred, average="weighted")
    f1_scores.append(f1)  # Store for visualization
    print(f"F1 score for {feature}: {f1:.2f}")

    # Track the best-performing feature
    feature_performance[feature] = f1
    if f1 > highest_f1:
        highest_f1 = f1
        best_feature_name = feature

# Store the best feature and its F1 score
best_predictive_feature = {best_feature_name: highest_f1}
print("\nBest predictive feature:", best_predictive_feature)

# Train and evaluate the model with all features combined
print("\nTraining with all features combined...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(multi_class="multinomial")
model.fit(X_train_scaled, y_train)

y_pred_combined = model.predict(X_test_scaled)
combined_f1 = f1_score(y_test, y_pred_combined, average="weighted")
print(f"F1 score using all features combined: {combined_f1:.2f}")

# Save results to a text file
with open("results_f1.txt", "w") as f:
    f.write(f"Best Predictive Feature: {best_predictive_feature}\n")
    for feature, score in feature_performance.items():
        f.write(f"{feature}: {score:.2f}\n")
    f.write(f"\nF1 score using all features combined: {combined_f1:.2f}\n")
print("Results saved to results_f1.txt")

# Visualize the F1 scores for individual features
plt.bar(features, f1_scores, color="skyblue")
plt.xlabel("Features")
plt.ylabel("F1 Score")
plt.title("F1 Scores for Individual Features")
plt.show()
