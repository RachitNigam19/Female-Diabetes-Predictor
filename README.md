# 🩺 Female-Diabetes-Predictor
This repository contains a machine learning-powered application for predicting diabetes in females based on medical data. Built with a web interface (using Streamlit or Flask) and scikit-learn for predictive modeling, this project demonstrates expertise in data science, web development, and containerized development environments.
📖 Overview
The Female Diabetes Predictor uses a trained machine learning model to classify diabetes risk in females based on features like glucose levels, BMI, or age. The project includes a Jupyter Notebook (diabetes prediction.ipynb) for model development, a web app (Diabetes Prediction web app.py) for user interaction, and a dataset (diabetes.csv). It also supports containerized development with a .devcontainer configuration for reproducibility.
🎯 Features

Predicts diabetes risk in females using a trained ML model.
Interactive web interface for inputting medical data and viewing predictions.
Includes data preprocessing and model training in a Jupyter Notebook.
Supports containerized development with .devcontainer for consistent environments.
Uses a real-world dataset (diabetes.csv) for accurate predictions.
Modular codebase for easy maintenance and scalability.

🛠️ Tech Stack

Python: Core programming language.
Streamlit/Flask: For building the interactive web interface (assumed from Diabetes Prediction web app.py).
Scikit-learn: For training and serving the diabetes prediction model.
Pandas/NumPy: For data manipulation and preprocessing.
Jupyter Notebook: For interactive model development and analysis.
Conda: For environment management (via environment.yml).
Git: Version control with .gitignore for clean repository management.
Dev Containers: For reproducible development environments (via .devcontainer).

🚀 Getting Started
Prerequisites

Python 3.8+
Jupyter Notebook or JupyterLab
Git
Docker (optional, for .devcontainer usage)
Conda (optional, for environment.yml)

Installation

Clone the repository:git clone https://github.com/RachitNigam19/Female-Diabetes-Predictor.git
cd Female-Diabetes-Predictor


Option 1: Install dependencies using requirements.txt:pip install -r requirements.txt


Option 2: Set up a Conda environment using environment.yml:conda env create -f environment.yml
conda activate diabetes-env


Ensure the serialized model (srained_model.sav) and dataset (diabetes.csv) are in the root directory.

Usage

Run the web application:python Diabetes Prediction web app.py


If using Streamlit, access at http://localhost:8501.
If using Flask, access at http://localhost:5000.


Input medical data (e.g., glucose, BMI) via the web UI to get diabetes predictions.
Explore the Jupyter Notebook (diabetes prediction.ipynb) for model training and evaluation:jupyter notebook diabetes prediction.ipynb



Dev Container Setup (Optional)

Open the repository in VS Code with the Dev Containers extension.
Reopen in container using the .devcontainer configuration for a preconfigured development environment.

📂 Project Structure
Female-Diabetes-Predictor/
├── .devcontainer/               # Dev container configuration
├── Diabetes Prediction web app.py  # Main web application (Streamlit or Flask)
├── diabetes prediction.ipynb    # Jupyter Notebook for model development
├── diabetes.csv                 # Dataset for training and predictions
├── predictive system.py         # Prediction logic for the model
├── srained_model.sav           # Serialized machine learning model
├── requirements.txt             # Project dependencies (pip)
├── environment.yml              # Conda environment configuration
├── .gitignore                   # Files/folders to ignore in Git

🔍 How It Works

Dataset: The diabetes.csv file contains a dataset of female medical data (e.g., glucose, BMI, age) for training the model.
Model: The srained_model.sav file stores a trained scikit-learn model (e.g., logistic regression or SVM) for predicting diabetes risk.
Preprocessing: The diabetes prediction.ipynb notebook handles data preprocessing, feature engineering, and model training.
Prediction Logic: The predictive system.py script likely implements the prediction pipeline for the web app.
Web UI: The Diabetes Prediction web app.py script serves a web interface for users to input data and view predictions.

🌟 Why This Project?

Demonstrates expertise in machine learning for health-related applications.
Showcases skills in building interactive web applications with Streamlit or Flask.
Highlights proficiency in data preprocessing and model development.
Reflects DevOps knowledge with containerized development environments.
Provides a practical example of an ML-driven tool for medical diagnostics.

📫 Contact

GitHub: RachitNigam19
LinkedIn: Rachit Nigam
Email: rachitn46@gmail.com

Feel free to explore, contribute, or reach out for collaboration!
