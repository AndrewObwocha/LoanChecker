# Loan Default Prediction Pipeline

## Overview

This Python script (`loan_checker.py`) implements an end-to-end machine learning pipeline to predict the likelihood of loan default based on borrower data. It reads data from CSV-like files, performs cleaning, visualization, class balancing, feature engineering, trains a Decision Tree classifier, evaluates the model, predicts outcomes for new loan requests, and presents the results in an interactive command-line interface.

The pipeline demonstrates common steps in a data science workflow, including data preprocessing, exploratory visualization, model building, and basic deployment via a CLI.

## Features

*   **Data Loading:** Reads borrower data from specified CSV-like files (`credit_risk_train.csv`, `loan_requests.csv`).
*   **Data Cleaning:** Removes records with missing values or unrealistic age entries (>= 90). Reports counts of removed records and missing values per column.
*   **Data Visualization:** Generates plots using `matplotlib` to explore:
    *   Age distribution of defaulters vs. non-defaulters (Histograms).
    *   Home ownership status among defaulters vs. non-defaulters (Pie Chart).
*   **Class Balancing:** Addresses class imbalance in the training data by performing simple undersampling of the majority class (non-defaulters).
*   **Feature Engineering & Selection:** Selects specific features (`loan_amnt`, `person_income`, `cb_person_cred_hist_length`) and scales numerical features using `StandardScaler`.
*   **Model Training:** Trains a `DecisionTreeClassifier` using `scikit-learn` on the prepared training data.
*   **Model Evaluation:** Assesses the trained model's performance on a held-out test set using:
    *   Accuracy Score.
    *   Classification Report (Precision, Recall, F1-score).
    *   Confusion Matrix.
*   **Prediction:** Uses the trained model to predict default status for new loan requests from `loan_requests.csv`.
*   **Interactive Display:** Presents the borrower details and predictions using a custom `Carousel` class, allowing the user to navigate back and forth through the records via the command line.

## Input Data Files (Required)

This script **requires** the following files to be present in the **same directory**:

1.  **`credit_risk_train.csv`**: Contains the historical training data with borrower information and known loan outcomes (`loan_status`). Expected to be comma-separated with a header row.
2.  **`loan_requests.csv`**: Contains new loan applicant data for prediction. Expected to be comma-separated with a header row similar to the training data (excluding `loan_status`).
3.  **`carousel.py`**: Contains the definition for the `Carousel` class used in the interactive display.

## Requirements

*   Python 3.x
*   `matplotlib`
*   `scikit-learn`
*   The custom `carousel.py` file.

## Installation

1.  **Place Files:** Ensure `loan_checker.py`, `credit_risk_train.csv`, `loan_requests.csv`, and `carousel.py` are in the same directory.
2.  **Install Libraries:** Open your terminal or command prompt and run:
    ```bash
    pip install matplotlib scikit-learn
    # or pip3 install matplotlib scikit-learn
    ```

## Usage

1.  **Navigate:** Open your terminal or command prompt and navigate to the directory containing all the required files.
2.  **Run the script:**
    ```bash
    python loan_checker.py
    # or python3 loan_checker.py
    ```
3.  **Observe Output:**
    *   The script will first print logs related to data cleaning, balancing, and model evaluation metrics.
    *   Plots generated during the visualization step will be displayed sequentially. You may need to close each plot window to proceed.
    *   Predictions for borrowers in `loan_requests.csv` will be printed.
    *   You will be prompted to press Enter to start the interactive carousel display.
4.  **Interact with Carousel:**
    *   Use `1` to move to the next borrower, `2` to move to the previous borrower, and `0` to exit the carousel interface.

## Code Structure

The script is organized into functions responsible for different pipeline stages:

*   `createDataFrame()`: Loads data.
*   `dataCleaning()`: Cleans the training data.
*   `dataVisualisation()`: Generates plots.
*   `classBalancing()`: Undersamples the majority class.
*   `featureSelection()`: Selects and scales features.
*   `modelTraining()`: Trains the Decision Tree.
*   `modelEvaluation()`: Evaluates the model.
*   `borrowerPrediction()`: Predicts on new data and populates the carousel.
*   `displayBorrower()`: Formats and prints the current borrower's info.
*   `clear()`: Clears the console screen.
*   `interface()`: Handles user interaction with the carousel.
*   `main()`: Orchestrates the execution of the entire pipeline.

## License

MIT License

## Author

Andrew Obwocha
