# Import libraries
import html
from IPython.display import display
import matplotlib.pyplot as plt
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import string


def get_drug_names(data):
    """
    Creates a list of unique drugNames
    
    data: Input dataframe
    return: List of unique drugs
    """
    
    # Create list of unique drugs
    drug_names = data['drugName'].unique()

    # Splits doubled drugs separated by ' / ', EG, 'Buprenorphine / naloxone'
    split_names = []
    
    for name in drug_names:
        split_names.extend(name.split(' / '))

    return split_names


def remove_drugNames(review_text, drugs):
    """
    Remove drugNames from reviews so the Q3 model won't be told directly what the drug being reviewed is
    
    review_text: Instance's Review
    drugs: List of drugNames from get_drug_names()
    return: Cleaned instance
    """

    # Regex pattern to match any of the drugNames from get_drug_names()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, drugs)) + r')\b', re.IGNORECASE)
    
    # Replace matched drug names with an empty string
    review_cleaned = pattern.sub('', review_text)

    # Strip extra spaces and return cleaned review
    return review_cleaned.strip()


def clean_data(data):
    """
    Remove duplicate values, nulls, and invalid instances; select columns needed for processing.
    Label conversion and X, y splitting occur in subsequent functions.

    data: Input dataframe
    return: Cleaned dataframe
    """

    print("\n********** Load Data ********************************\n")
    
    # Check number of rows and columns
    print("No. of Rows: {}".format(data.shape[0]))
    print("No. of Columns: {}".format(data.shape[1]))
    
    print("\n********** Data Cleaning *****************************")
     
    # Render HTML characters
    data.loc[:, 'review'] = data.loc[:, 'review'].apply(html.unescape)

    # Replace new lines and tabs with spaces
    data.loc[:, 'review'] = data.loc[:, 'review'].str.replace('\n', ' ').str.replace('\t', ' ')
    
    # Create a list of unique drugNames
    drug_names = get_drug_names(data)
    
    # Trim unnecessary spaces from strings
    data.loc[:, "drugName"] = data.loc[:, "drugName"].apply(lambda x: str(x).strip())
    data.loc[:, "condition"] = data.loc[:, "condition"].apply(lambda x: str(x).strip())
    data.loc[:, "review"] = data.loc[:, "review"].apply(lambda x: str(x).strip())

    # Remove generic/brand name duplicate reviews, keeping first instance
    ## Alphebetize by drugName so that the same variant is always removed for the same drug
    data = data.sort_values(by='drugName')
    data.drop_duplicates(subset=['review', 'date', 'rating'], keep='first', inplace=True)
    print("\nNo. of rows (After removing duplicates): {}".format(data.shape[0]))
    
    # Remove null values
    data.replace('nan', np.nan, inplace=True)
    data.dropna(inplace=True)
    print("\nNo. of rows (After dropping null): {}".format(data.shape[0]))
    
    # Remove instances with the usefulCount copied into the Condition
    ## Affected instances all start with a digit, E.G., "12</span> users found this comment helpful."
    data = data[~data['condition'].str.match(r'^\d')]
    print("\nNo. of rows (After dropping invalid Conditions): {}".format(data.shape[0]))
    
    
    # Remove any drugNames from review text to not spoil Q3
    print("\n********** Cleaning Review Data... *********************\n")
    data['cleaned_review'] = data['review'].apply(lambda review: remove_drugNames(review, drug_names))
    
    # Select data needed for processing
    data = data[['drugName', 'condition', 'review', 'cleaned_review', 'rating']]
    print("No. of columns used for processing: {}".format(data.shape[1]))
    
    print('\n<Data View after Cleaning: First Few Instances>')
    display(data.head(5))

    return data


def convert_labels(data, target_drug):
    """
    Convert label to target class for each research question and rename target class column to Label
    
    data: Cleaned dataframe
    return: X, y dataframes for use with each research question 1-3
    """
    # Load model from backup so they aren't learning from other predicts
    q1_data = data.copy(deep=True)
    q3_data = data.copy(deep=True)
    
    # Q1 & Q2. Convert label (rating) to target class
    ## Negative class: Rating 1-5
    ## Positive class: Rating 6-10
    q1_data.loc[:, "rating"] = q1_data.loc[:, "rating"].apply(lambda x: 1 if x >= 6 else 0)
    
    ## Select necessary columns and rename as necessary
    q1_data = q1_data[['drugName', 'condition', 'review', 'rating']]
    q1_data.rename(columns={"rating": "label"}, inplace=True)
    
    print("********** Label Distribution of Target Classes ************")
    
    print('\nQ1. Class Counts (label, row):')
    print(q1_data["label"].value_counts())
    print("\nQ1. Data View: X Data")
    display(q1_data.head(3))
    
    # Q3. Convert label (drugName) to target class
    q3_data.loc[:, "drugName"] = q3_data.loc[:, "drugName"].apply(lambda x: 1 if x == target_drug else 0)
    
    ## Select necessary columns and rename as necessary
    q3_data = q3_data[['rating', 'cleaned_review', 'drugName']]
    q3_data.rename(columns={"cleaned_review": "review"}, inplace=True)
    q3_data.rename(columns={"drugName": "label"}, inplace=True)
    
    print('\n\nQ3. Class Counts (label, row):')
    print(q3_data["label"].value_counts())
    print("\nQ3. Data View: X Train")
    display(q3_data.head(3))
    
    return q1_data, q3_data


def split_data(data):
    """
    Split dataset into X and y dataframes with subsets for train (75%) and test (25%).
    
    data: Dataframe obtained from clean_data()
    return: X_train, X_test, y_train, y_test
    
    """
    
    print("\n********** Splitting Data *****************************\n")
    
    # 1. Split into X and y (target)
    X_data, y_data = data.iloc[:, :-1], data.iloc[:, -1]
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=28, stratify=y_data)
    
    # 3. Reset indices

    ## Train Data
    X_train=X_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)

    ## Test Data
    X_test=X_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    
    # 4. Report Results
    print("Train Data: ", X_train.shape)
    print("Test Data: ", X_test.shape)
    
    print("\nClass Counts (label, row): Train\n", y_train.value_counts())
    print("\nClass Counts (label, row): Test\n", y_test.value_counts())
    
    print("\nData View: X Train")
    display(X_train.head(3))
    print("\nData View: X Test")
    display(X_test.head(3))
    
    # Re-merge X_test and y_test for Q2
    q2_data = X_test.copy(deep=True)
    q2_data['label'] = y_test
    
    return (X_train, X_test, y_train, y_test, q2_data)


def preprocess_data(data):
    """
    Preprocess data by joining columns, applying lowercase conversion, punctuation removal, tokenization, stemming, and removing unnecessary spaces

    data: dataframe received from split_data()
    return: transformed dataframe
    
    """
    
    print("\n********** Preprocessing Data... ***********************\n")
    
    # Joins text columns for processing
    X_data = data.apply(lambda row: ' '.join(row.astype(str)), axis=1)

    # 1. convert all characters to lowercase
    X_data = X_data.map(lambda x: x.lower())
    
    # 2. remove punctuation
    X_data = X_data.str.replace(f"[{string.punctuation}]", "", regex=True)
    
    # 3. tokenize sentence
    X_data = X_data.apply(nltk.word_tokenize)

    # 4. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])

    # 5. Trim unnecessary spaces
    X_data.apply(lambda x: str(x).strip())
    
    # 6. convert to string of tokens separated by a space
    X_data = X_data.apply(lambda x: " ".join(x))

    print("X Data Shape:", X_data.shape)
    print("\nData View: X Data")
    display(X_data.head(3))
    
    return X_data


def fit_model(X, y):
    """
    Perform model fitting on preprocessed X and y dataframes

    X, y: Dataframes created by preprocess_data()
    return: Fitted model and vectorizer
    """
    
    print("\n********** Training Model... ***************************")
    # Vectorizes text data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7)
    X = vectorizer.fit_transform(X)
    
    # Trains model
    model = LogisticRegressionCV(cv=10, class_weight='balanced', scoring='f1_weighted', random_state=28, max_iter=1000)
    
    # Ensures lables are ints. Required for Q3
    y = y.astype(int)

    # Fits model
    model = model.fit(X, y)
    
    # Backup file to prevent contamination between evaluations
    with open('lr_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model, vectorizer


def evaluate_model(predicted_labels, actual_labels, model, message):
    """
    Evaluates the model generated by fit_model() to produce a confusion matrix
    
    predicted_labels: Predictions from model
    actual_labels: y dataframe obtained from split_data()
    model: Trained model from fit_model()
    return: None; displays the confusion matrix and evaluation metrics
    """
    
    print("\n********** Model Evaluation...**************************")
    
    print(f'\n{message} Confusion Matrix:\n')
    
    # Makes sure both sets of lables are ints
    predicted_labels = predicted_labels.astype(int)
    actual_labels = actual_labels.astype(int)
    
    # Display a confusion matrix with colors and label names
    matrix = confusion_matrix(actual_labels, predicted_labels)
    confusion_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    plt.figure(figsize=(2, 2))
    confusion_display.plot(cmap='bwr', values_format='d', ax=plt.gca())
    for text in confusion_display.ax_.texts:
        text.set_fontsize(12)
    
    plt.show()
    
    # Displays a classification report
    print(f'\n{message} Classification Report:\n')
    report_dict = classification_report(actual_labels, predicted_labels, digits=3, output_dict=True)
    display(pd.DataFrame(report_dict).transpose())
    
    
def split_test_by_class(test_data):
    """
    Separate out one condition from the others
    
    test_data: Dataframe of all conditions
    Return: X, y dataframes for depression, non-depression, and birth control.
    """
    
    print("\n********** Splitting Data by Class **********************\n")
    
    # 1. Separate depression instances from other conditions
    ## Searching for 'depress' in order to return 'Depression', 'Major Depressive Disorder', 'Postpartum Depression', etc.
    depression_data = test_data[test_data['condition'].str.contains('depress', case=False)]
    other_data = test_data[~test_data['condition'].str.contains('depress', case=False)]
    birth_control_data = test_data[test_data['condition'].str.contains('Birth Control', case=False)]
    
    # 2. Split into X and y (target)
    X_data_depression, y_data_depression = depression_data.iloc[:, :-1], depression_data.iloc[:, -1]
    X_data_other, y_data_other = other_data.iloc[:, :-1], other_data.iloc[:, -1]
    X_data_birth_control, y_data_birth_control = birth_control_data.iloc[:, :-1], birth_control_data.iloc[:, -1]
   
    # 3. Reset indices

    ## Depression Data
    X_data_depression=X_data_depression.reset_index(drop=True)
    y_data_depression=y_data_depression.reset_index(drop=True)

    ## Non-Depression Data
    X_data_other=X_data_other.reset_index(drop=True)
    y_data_other=y_data_other.reset_index(drop=True)
    
    ## Birth Control Data
    X_data_birth_control=X_data_birth_control.reset_index(drop=True)
    y_data_birth_control=y_data_birth_control.reset_index(drop=True)
    
    # 4. Report results
    print("Depression Data: ", X_data_depression.shape)
    print("Non-Depression Data: ", X_data_other.shape)
    print("Birth Control Data: ", X_data_birth_control.shape)
    
    print("\nClass Counts (label, row): Depression\n", y_data_depression.value_counts())
    print("\nClass Counts (label, row): Non-Depression\n", y_data_other.value_counts())
    print("\nClass Counts (label, row): Birth Control\n", y_data_birth_control.value_counts())
    
    print("\nData View: X Depression")
    display(X_data_depression.head(3))
    print("\nData View: X Non-Depression")
    display(X_data_other.head(3))
    print("\nData View: X Birth Control")
    display(X_data_birth_control.head(3))
    
    return (X_data_depression, y_data_depression, X_data_other, y_data_other, X_data_birth_control, y_data_birth_control)


def load_model():
    """
    Loads fresh model from saved file
    
    return: Model with no knowledge of Test data
    """
    
    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    return model
