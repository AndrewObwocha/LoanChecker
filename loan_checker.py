'''
Loan Default Prediction System

This module implements a complete machine learning pipeline for analyzing and predicting 
whether a loan applicant will default on their loan. The pipeline includes:

    - Loading and parsing borrower data from a file
    - Cleaning the dataset by removing incomplete or irrelevant records
    - Visualizing demographic and loan-related trends among defaulters and non-defaulters
    - Balancing the dataset to reduce class imbalance
    - Extracting and scaling predictive features
    - Training a decision tree classifier to predict loan defaults
    - Evaluating model performance using standard classification metrics
    - Generating predictions for new borrowers
    - Providing a command-line interface for browsing predictions

Author: Andrew Obwocha
Date: 5th April 2025
'''

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from carousel import Carousel
import os  # For clearing command-line


def createDataFrame(filename):
    '''
    Parses a CSV file and constructs a list of dictionaries representing rows of data

    Parameters:
        filename (str): The name of the CSV file containing data rows

    Returns:
        list[dict]: A list of dictionaries where each dictionary corresponds to a row of data.
                    Keys are column names, and values are the corresponding entries as strings
    '''
    with open(filename, 'r') as file:
        dataRows = file.read().splitlines()
    
    dataRows = [row.strip().split(',') for row in dataRows]
    titleRow = dataRows[0]  # First row contains column headers
    
    dataFrame = []
    for row in dataRows[1:]:
        # Loop through the remaining rows to build a dictionary for each
        rowData = {}
        for index, title in enumerate(titleRow):
            rowData[title] =  row[index]
        dataFrame.append(rowData)
            
    return dataFrame


def dataCleaning(dataframe):
    '''
    Cleans the input dataset by removing rows with missing values or overage entries

    Parameters:
        dataframe (list[dict]): A list of dictionaries where each dictionary corresponds to a row of data. 
                                Keys are column names, and values are the corresponding entries as strings

    Returns:
        None: The input list is modified in-place. The function prints out:
              - Number of rows removed due to missing data or age >= 90
              - Count of missing values per column
    '''   
    print(f'Initial number of rows: {len(dataframe)}')
    
    columnEmptyValues = {}
    overAgeRows = 0

    # Iterate over the dataframe in reverse to safely delete rows in-place
    for index in range(len(dataframe) - 1, -1, -1):
        # Delete row if any of its values are missing
        deletedRow = False
        for key, value in dataframe[index].items():
            if not value:
                try:
                    columnEmptyValues[key] += 1
                except KeyError:
                    columnEmptyValues[key] = 1
                del dataframe[index]
                deletedRow = True
        
        # If the row wasn't deleted for missing values, check the age condition
        if not deletedRow and int(dataframe[index]['person_age']) >= 90:
            overAgeRows += 1
            del dataframe[index]

    # Report which columns (if any) had missing values and how many
    if columnEmptyValues:
        for key, value in columnEmptyValues.items():
            print(f'Column {key}: {value} values missing')

    numberOfRows = len(dataframe)
    print(f'Remaining number of rows: {numberOfRows + overAgeRows}')
    
    print(f'\nNumber of records with age > 90: {overAgeRows}')
    print(f'Remaining number of rows: {numberOfRows}')
    

def dataVisualisation(dataframe):
    '''
    Generates visualizations to explore patterns in loan default data

    Parameters:
        dataframe (list[dict]): A list of dictionaries where each dictionary corresponds to a row of data. 
                                Keys are column names, and values are the corresponding entries as strings

    Returns:
        None: Displays three plots:
            - Histogram of ages of borrowers who defaulted
            - Histogram of ages of borrowers who did not default
            - Pie chart of home owners who defaulted vs. those who didn't

    '''
    defaultersAges = []
    nonDefaultersAges = []
    
    defaultedHomeOwners = 0
    nonDefaultedHomeOwners = 0
    
    # Categorize data by default status and home ownership
    for row in dataframe:
        defaulted = False
        if row['loan_status'] == '1':
            defaultersAges.append(int(row['person_age']))
            defaulted = True
        else:
            nonDefaultersAges.append(int(row['person_age']))

        # Count homeowners by default status
        if row['person_home_ownership'] == 'OWN':
            if defaulted:
                defaultedHomeOwners += 1
            else:
                nonDefaultedHomeOwners += 1

    # Histogram: Ages of people who defaulted
    plt.figure(figsize=(8, 6))
    plt.hist(defaultersAges, bins=10)
    plt.title('Loans in Default')
    plt.xlabel('Age (in years)')
    plt.ylabel('No. of Borrowers')
    plt.show()

    # Histogram: Ages of people who did not default
    plt.figure(figsize=(8, 6))
    plt.hist(nonDefaultersAges, bins=10)
    plt.title('Loans not in Default')
    plt.xlabel('Age (in years)')
    plt.ylabel('No. of Borrowers')
    plt.show()

    # Pie chart: Ownership status among defaulters vs non-defaulters
    categorySizes = [defaultedHomeOwners, nonDefaultedHomeOwners]
    categoryLabels = ['Defaulted', 'Not Defaulted']    

    plt.figure(figsize=(8, 6))
    plt.pie(categorySizes, labels=categoryLabels, autopct='%1.1f%%')
    plt.title('Home Owners: Default vs. Not Default')
    plt.show()


def classBalancing(dataframe):
    '''
    Balances the dataset by undersampling the majority class (non-defaulters)

    Parameters:
        dataframe (list[dict]): A list of dictionaries where each dictionary corresponds to a row of data. 
                                Keys are column names, and values are the corresponding entries as strings

    Returns:
        None: Modifies the input list in-place by deleting excess non-defaulting entries.
              Prints the number of defaulters and non-defaulters before and after balancing
    '''
    # Collect indices of non-defaulters in reverse to safely delete them later
    nonDefaultersIndices = [index for index in range(len(dataframe) - 1, -1, -1) if dataframe[index]['loan_status'] == '0']
    
    numOfNonDefaulters = len(nonDefaultersIndices)
    numOfDefaulters = len(dataframe) - numOfNonDefaulters
    print('\nPre-balancing')
    print(f'Number of borrowers who defaulted: {numOfDefaulters}')
    print(f'Number of borrowers who did not default: {numOfNonDefaulters}')

    for index in nonDefaultersIndices[numOfDefaulters:]:
        del dataframe[index]

    # Re-count non-defaulters again after deletion
    numOfNonDefaulters = sum([1 for index in range(len(dataframe)) if dataframe[index]['loan_status'] == '0'])
    numOfDefaulters = len(dataframe) - numOfNonDefaulters
    print('\nPost-balancing')
    print(f'Number of borrowers who defaulted: {numOfDefaulters}')
    print(f'Number of borrowers who did not default: {numOfNonDefaulters}')
    

def featureSelection(dataframe):
    '''
    Prepares feature vectors and labels for model training

    Parameters:
        dataframe (list[dict]): A list of dictionaries where each dictionary corresponds to a row of data. 
                                Keys are column names, and values are the corresponding entries as strings

    Returns:
        tuple:
            features (list[list[float]]): A 2D list where each inner list is a feature vector
            labels (list[int]): Corresponding target labels (0 or 1)
    '''
    features = []
    labels = []
    dataToScale = []
    
    for row in dataframe:
        features.append([int(row['cb_person_cred_hist_length'])])  # Store the raw features to be scaled
        dataToScale.append([int(row['loan_amnt']), int(row['person_income'])])  # Store the feature that doesn't need scaling
        labels.append(int(row['loan_status']))  # Store the label
    
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(dataToScale)  # Scale features

    # Concatenate scaled features with existing feature
    for index in range(len(features)):
        features[index] = list(scaledData[index]) + features[index]

    return features, labels


def modelTraining(features, labels):
    '''
    Trains a Decision Tree classifier on the given features and labels

    Parameters:
        features (list[list[float]]): A 2D list where each inner list is a feature vector
        labels (list[int]): A list of binary labels corresponding to each feature vector (0 or 1)

    Returns:
        tuple:
            - clf (DecisionTreeClassifier): The trained decision tree model
            - X_test (list[list[float]]): Feature vectors reserved for testing
            - y_test (list[int]): Labels corresponding to the test set
    '''
    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test    


def modelEvaluation(model, X_test, y_test):
    '''
    Evaluates a trained classification model on test data and prints key performance metrics

    Parameters:
        model: A trained classification model
        X_test (list[list[float]]): Feature vectors for the test set
        y_test (list[int]): True class labels for the test set

    Returns:
        None: Prints:
            - Test accuracy (proportion of correct predictions)
            - Classification report (precision, recall, F1-score per class)
            - Confusion matrix
    '''
    # Get predictions from the trained model
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print(f'\nTest Accuracy: {accuracy_score(y_test, y_pred):.2f}')  # Accuracy
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')  # Precision, Recall, F1-score
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')  # Confusion Matrix


def borrowerPrediction(model, borrowerDataFrame):
    '''
    Uses a trained model to predict loan default risk for new borrower records.

    Parameters:
        model: A trained classification model
        borrowerDataFrame (list[dict]): A list of dictionaries where each dictionary corresponds to a row of data. 
                                        Keys are column names, and values are the corresponding entries as strings

    Returns:
        list[dict]: The updated borrowerDataFrame, each with an added 'prediction' key.
    '''
    borrowerFeatures = []
    borrowerDataToScale = []
    
    for borrower in borrowerDataFrame:
        borrowerFeatures.append([int(borrower['cb_person_cred_hist_length'])])
        borrowerDataToScale.append([int(borrower['loan_amnt']), int(borrower['person_income'])])
    
    scaler = StandardScaler()
    borrowerScaledData = scaler.fit_transform(borrowerDataToScale)  # Scale features

    # Concatenate scaled features with existing feature
    for index in range(len(borrowerFeatures)):
        borrowerFeatures[index] = list(borrowerScaledData[index]) + borrowerFeatures[index]

    borrowerPredictions = model.predict(borrowerFeatures)  # Make predictions
    
    # Add borrower records to circular-queue
    carousel = Carousel()
    print('\nBorrwer predicitons:\n')
    for index, borrower in enumerate(borrowerDataFrame):
        borrower['prediction'] = borrowerPredictions[index]
        print(f'{borrower['borrower']}: {borrower['prediction']}')
        carousel.add(borrower)

    return carousel 
        

def displayBorrower(carousel):
    '''
    Displays a formatted summary of the current borrower's loan application and prediction.

    Parameters:
        carousel (CircularQueue): A CircularQueue object that stores the records of borrowers

    Returns:
        None: Prints: A formatted report including borrower details and a prediction-based recommendation.

    '''
    titles = [
        'Borrower', 
        'Age', 
        'Income', 
        'Home_ownership', 
        'Employment', 
        'Loan intent', 
        'Loan grade', 
        'Amount', 
        'Interest Rate', 
        'Loan percent income',
        'Historical Defaults',
        'Credit History',
        'Predicted loan_status'
        ]
    
    currentBorrower = carousel.getCurrentData()
    print('-'*50)
    for index, key in enumerate(currentBorrower.keys()):
        # Custom display logic
        if key in ['person_income', 'loan_amnt']:
            print(f'{titles[index]}: ${currentBorrower[key]}\n')
        elif key == 'cb_person_default_on_file':
            print(f'{titles[index]}: {'Yes' if currentBorrower[key] == 'Y' else 'No'}\n')
        elif key == 'cb_person_cred_hist_length':
            print(f'{titles[index]}: {currentBorrower[key]} years\n')
        elif key == 'prediction':
            print('-'*50)
            print(f'\n{titles[index]}: {'Will default' if currentBorrower[key] == 1 else 'Will not default'}')
            print(f'Recommend: {'Deny' if currentBorrower[key] == 1 else 'Accept'}\n')
            print('-'*50)
        else:
            print(f'{titles[index]}: {currentBorrower[key]}\n')


def clear():
    '''
    Clears the screen based on the operating system.
    '''
    if os.name == "posix":
        os.system('clear')
    else:
        os.system('cls')


def interface(carousel):
    '''
    Interface that allows the user to browse through borrower records and their loan default predictions

    Parameters:
        carousel (CircularQueue): A CircularQueue object that stores the records of borrowers

    Returns:
        None
    '''    
    input("\nPress enter key to begin carousel display...")
    clear()
    displayBorrower(carousel)

    carouselRunning = True
    while carouselRunning:
        validChoice = False
        while not validChoice:
            userChoiceStr = input('\nChoose one of the following options:\n\t1. Go next\n\t2. Go back\n\t0. Exit\n>>>> ')
            try:
                userChoice = int(userChoiceStr)
                if userChoice in range(0, 3):
                    validChoice = True
                else:
                    print("Invalid entry. Number must be 0, 1, or 2.")
            except ValueError:
                print("Invalid entry. Number must be 0, 1, or 2.")
        
        if userChoice == 0:
            carouselRunning = False
        else:
            if userChoice == 1:
                carousel.moveNext()
            elif userChoice == 2:
                carousel.movePrevious()
            clear()
            displayBorrower(carousel)

    print('Goodbye!')
                     

def main():
    dataFrame = createDataFrame('credit_risk_train.csv')
    dataCleaning(dataFrame)
    dataVisualisation(dataFrame)
    classBalancing(dataFrame)
    
    features, labels = featureSelection(dataFrame)
    model, X_test, y_test = modelTraining(features, labels)
    modelEvaluation(model, X_test, y_test)

    borrowerDataFrame = createDataFrame('loan_requests.csv')
    carousel = borrowerPrediction(model, borrowerDataFrame)
    interface(carousel)


if __name__ == '__main__':
    main()
