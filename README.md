# Fraud Detection System
This is task 2 of VerveBridge [2024 June] <br>
**https://vervebridge-frauddection.onrender.com** <br>
This repository contains a complete pipeline for detecting fraudulent transactions using machine learning models. The project involves data preprocessing, model training, evaluation, and deployment of an ensemble model for fraud detection.

## Table of Contents

- Installation
- Usage
- Project Structure
- Model Training
- Evaluation
- Web Application
- Contributing
- License

## Installation

To run this project, you need to have Python installed on your machine. You can install the required packages by running:

```bash
pip install -r requirements.txt

```
#### 1 Usage
Data Preprocessing and Model Training:

Run main.py to load the dataset, preprocess it, train the models, and evaluate them.
```bash

python main.py
```
#### 2 Web Application:

Run server.py to start the Flask web application, which allows users to input transaction data and get predictions on whether they are fraudulent.
```bash

python server.py
```
#### 3 Logging:

Logs are stored in logger.log for tracking the process flow and debugging.

## Project Structure
```bash

.
├── bank_dataset.csv
├── main.py
├── model.py
├── server.py
├── requirements.txt
├── logger.log
├── categorical_mappings.pkl
├── trained_ensemble_model.pkl
├── templates
│   └── index.html
└── static
    └── style.css
```
bank_dataset.csv: The dataset containing transaction data.<br>
main.py: Main script to load data, preprocess it, train models, and evaluate them.<br>
model.py: Contains functions for data preprocessing, model training, evaluation, and utility functions.<br>
server.py: Flask application for serving the fraud detection model.<br>
requirements.txt: List of required Python packages.<br>
logger.log: Log file for tracking the process.<br>
categorical_mappings.pkl: Pickle file to store categorical mappings for preprocessing.<br>
trained_ensemble_model.pkl: Pickle file to store the trained ensemble model.<br>
templates/index.html: HTML template for the web application.<br>
static/style.css: CSS file for styling the web application.<br>
## Model Training
### 1 Load and Preprocess Data:

The load_data() function loads the dataset.
The preprocess_data() function processes the data by handling categorical features and splitting it into features (X) and target (y).
### 2Train Models:

Individual models are trained using K-Nearest Neighbors, Random Forest, and XGBoost.
An ensemble model is created and trained using a VotingClassifier that combines the predictions of the individual models.
### 3 Evaluate Models:

Models are evaluated using classification reports, confusion matrices, and ROC-AUC curves.
## Evaluation
The evaluate_model() function prints the classification report and confusion matrix. The plot_roc_auc() function plots the ROC-AUC curve for visual evaluation of the model performance.

## Web Application
A Flask web application is provided to input new transaction data and get predictions. The app is located in server.py and can be started by running:

```bash

python server.py
```
Navigate to http://0.0.0.0:5000 in your web browser to use the application.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
