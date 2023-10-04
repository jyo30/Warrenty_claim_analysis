import json
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the trained model and vectorizer from the files
model_filename = "supplier_validity_model.pkl"
loaded_model = joblib.load(model_filename)

vectorizer_filename = "count_vectorizer.pkl"
vectorizer = joblib.load(vectorizer_filename)

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load abbreviations
abbreviations_df = pd.read_excel('ABBREV.xlsx')
abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))

# Function to preprocess text
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [abbreviations[word] if word in abbreviations else word for word in words]
    return " ".join(words)

if __name__ == '__main__':
    # Load the condition values from the JSON file
    with open('a.json', 'r') as json_file:
        condition_values_list = json.load(json_file)

    # Read the Excel file into a DataFrame
    df = pd.read_excel('ST-GWC Debit ID 6001367_(DISPUTED CLAIMS) 05152023.xlsx')  # Replace with the actual file path

    # Extract the required columns and convert to datetime objects
    df['Warranty Start Date'] = pd.to_datetime(df['Warranty Start Date'])
    df['Repair Date'] = pd.to_datetime(df['Repair Date'])

    # Calculate 'time_in_service_years'
    df['time_in_service_years'] = (df['Repair Date'] - df['Warranty Start Date']).dt.days / 365.25  # Consider leap years

    # Handling null values in 'Technician Comments' and 'Customer Comments' columns
    df['Technician Comments'].fillna("ok", inplace=True)
    df['Customer Comments'].fillna("ok", inplace=True)

    # Preprocess the comments
    df['Technician Comments'] = df['Technician Comments'].apply(preprocess_text)
    df['Customer Comments'] = df['Customer Comments'].apply(preprocess_text)

    # Combine comments
    df['combined_comments'] = df['Technician Comments'] + " " + df['Customer Comments']

    # Loop through each set of conditions and apply them to the DataFrame
    for condition_values in condition_values_list:
        # Set conditions to update 'predicted validity1' column
        condition1 = (df['Mileage'] > condition_values['Mileage']) | (df['time_in_service_years'] > condition_values['time_in_service_years']) | (df['Vehicle Line AWS'].isin(condition_values['Vehicle Line AWS']))
        df.loc[condition1, 'predicted validity1'] = 'rejected'

        # Set conditions to update 'final validity' column based on part numbers
        condition2 = df['Part Num Base (Causal)'].isin(condition_values['Part Num Base (Causal)'])
        df.loc[condition2, 'final validity'] = 'rejected'

        # Vectorize the comments using the loaded vectorizer
        X_vectorized = vectorizer.transform(df['combined_comments'])

        # Make predictions using the model
        df['predicted_validity2'] = loaded_model.predict(X_vectorized)

        # Save the updated DataFrame to a new Excel file for each set of conditions
        output_filename = f"output_with_predictions_{condition_values['Mileage']}_{condition_values['time_in_service_years']}.xlsx"
        df.to_excel(output_filename, index=False)

    print("Predictions saved to output files.")
