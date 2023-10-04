# # import pandas as pd
# # import joblib
# # import nltk
# # from nltk.corpus import stopwords
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.linear_model import LogisticRegression

# # # Load the trained model and vectorizer from the files
# # model_filename = "supplier_validity_model.pkl"
# # loaded_model = joblib.load(model_filename)

# # vectorizer_filename = "count_vectorizer.pkl"
# # vectorizer = joblib.load(vectorizer_filename)

# # # Load stopwords
# # nltk.download('stopwords')
# # stop_words = set(stopwords.words('english'))

# # # Load abbreviations
# # abbreviations_df = pd.read_excel('ABBREV.xlsx')
# # abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))

# # # Function to preprocess text
# # def preprocess_text(text):
# #     words = nltk.word_tokenize(text.lower())
# #     words = [word for word in words if word.isalpha() and word not in stop_words]
# #     words = [abbreviations[word] if word in abbreviations else word for word in words]
# #     return " ".join(words)

# # if __name__ == '__main__':
# #     # Read the Excel file into a DataFrame
# #     df = pd.read_excel('ST-GWC Debit ID 6001363_(DISPUTED CLAIMS) 05122023.xlsx')  # Replace with the actual file path

# #     # Handling null values in 'Technician Comments' and 'Customer Comments' columns
# #     df['Technician Comments'].fillna("ok", inplace=True)
# #     df['Customer Comments'].fillna("ok", inplace=True)

# #     # Preprocess the comments
# #     df['Technician Comments'] = df['Technician Comments'].apply(preprocess_text)
# #     df['Customer Comments'] = df['Customer Comments'].apply(preprocess_text)

# #     # Combine comments
# #     df['combined_comments'] = df['Technician Comments'] + " " + df['Customer Comments']

# #     # Vectorize the comments using the loaded vectorizer
# #     X_vectorized = vectorizer.transform(df['combined_comments'])

# #     # Make predictions
# #     df['predicted_validity'] = loaded_model.predict(X_vectorized)

# #     # Save the updated DataFrame to a new Excel file
# #     output_filename = "output_with_predictions.xlsx"
# #     df.to_excel(output_filename, index=False)

# #     print("Predictions saved to 'output_with_predictions.xlsx'")













# import pandas as pd
# import joblib
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# # Load the trained model and vectorizer from the files
# model_filename = "supplier_validity_model.pkl"
# loaded_model = joblib.load(model_filename)

# vectorizer_filename = "count_vectorizer.pkl"
# vectorizer = joblib.load(vectorizer_filename)

# # Load stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Load abbreviations
# abbreviations_df = pd.read_excel('ABBREV.xlsx')
# abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))

# # Function to preprocess text
# def preprocess_text(text):
#     words = nltk.word_tokenize(text.lower())
#     words = [word for word in words if word.isalpha() and word not in stop_words]
#     words = [abbreviations[word] if word in abbreviations else word for word in words]
#     return " ".join(words)

# if __name__ == '__main__':
#     # Read the Excel file into a DataFrame
#     df1 = pd.read_excel('ST-GWC Debit ID 6001363_(DISPUTED CLAIMS) 05122023.xlsx')  # Replace with the actual file path

#     # Extract the required columns and convert to datetime objects
#     df = df1.iloc[:, :-4]
#     df['Warranty Start Date'] = pd.to_datetime(df['Warranty Start Date'])
#     df['Repair Date'] = pd.to_datetime(df['Repair Date'])

#     # Calculate 'time_in_service_years'
#     df['time_in_service_years'] = (df['Repair Date'] - df['Warranty Start Date']).dt.days / 365.25  # Consider leap years

#     # Handling null values in 'Technician Comments' and 'Customer Comments' columns
#     df['Technician Comments'].fillna("ok", inplace=True)
#     df['Customer Comments'].fillna("ok", inplace=True)

#     # Preprocess the comments
#     df['Technician Comments'] = df['Technician Comments'].apply(preprocess_text)
#     df['Customer Comments'] = df['Customer Comments'].apply(preprocess_text)

#     # Combine comments
#     df['combined_comments'] = df['Technician Comments'] + " " + df['Customer Comments']

#     # Vectorize the comments using the loaded vectorizer
#     X_vectorized = vectorizer.transform(df['combined_comments'])

#     # Make predictions using the model
#     df['predicted_validity2'] = loaded_model.predict(X_vectorized)

#     # Add the 'predicted validity1' column with initial value 'Accepted'
#     df['predicted validity1'] = 'Accepted'

#     # Set conditions to update 'predicted validity1' column
#     condition = (df['Mileage'] > 36000) | (df['time_in_service_years'] > 3) | (df['Vehicle Line AWS'].isin(['M4', 'R2', 'ZE']))
#     df.loc[condition, 'predicted validity1'] = 'Rejected'

#     # Save the updated DataFrame to the same Excel file
#     output_filename = "output_with_predictions.xlsx"
#     df.to_excel(output_filename, index=False)

#     print("Predictions saved to 'output_with_predictions.xlsx'")









###################  reading excel file as input and getting the excel file as output #####
# import pandas as pd
# import joblib
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# # Load the trained model and vectorizer from the files
# model_filename = "supplier_validity_model.pkl"
# loaded_model = joblib.load(model_filename)

# vectorizer_filename = "count_vectorizer.pkl"
# vectorizer = joblib.load(vectorizer_filename)

# # Load stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Load abbreviations
# abbreviations_df = pd.read_excel('ABBREV.xlsx')
# abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))

# # Function to preprocess text
# def preprocess_text(text):
#     words = nltk.word_tokenize(text.lower())
#     words = [word for word in words if word.isalpha() and word not in stop_words]
#     words = [abbreviations[word] if word in abbreviations else word for word in words]
#     return " ".join(words)

# if __name__ == '__main__':
#     # Read the Excel file into a DataFrame
#     df = pd.read_excel('ST-GWC Debit ID 6001367_(DISPUTED CLAIMS) 05152023.xlsx')  # Replace with the actual file path

#     # Extract the required columns and convert to datetime objects
#     #df = df1.iloc[:, :-4]
#     df['Warranty Start Date'] = pd.to_datetime(df['Warranty Start Date'])
#     df['Repair Date'] = pd.to_datetime(df['Repair Date'])

#     # Calculate 'time_in_service_years'
#     df['time_in_service_years'] = (df['Repair Date'] - df['Warranty Start Date']).dt.days / 365.25  # Consider leap years

#     # Handling null values in 'Technician Comments' and 'Customer Comments' columns
#     df['Technician Comments'].fillna("ok", inplace=True)
#     df['Customer Comments'].fillna("ok", inplace=True)

#     # Preprocess the comments
#     df['Technician Comments'] = df['Technician Comments'].apply(preprocess_text)
#     df['Customer Comments'] = df['Customer Comments'].apply(preprocess_text)

#     # Combine comments
#     df['combined_comments'] = df['Technician Comments'] + " " + df['Customer Comments']

#     # Vectorize the comments using the loaded vectorizer
#     X_vectorized = vectorizer.transform(df['combined_comments'])

#     # Make predictions using the model
#     df['predicted_validity2'] = loaded_model.predict(X_vectorized)

#     # Add the 'predicted validity1' column with initial value 'Accepted'
#     df['predicted validity1'] = 'accepted'

#     # Set conditions to update 'predicted validity1' column
#     condition1 = (df['Mileage'] > 36000) | (df['time_in_service_years'] > 3) | (df['Vehicle Line AWS'].isin(['M4', 'R2', 'ZE']))
#     df.loc[condition1, 'predicted validity1'] = 'rejected'

#     # Add the 'final validity' column based on the conditions
#     df['final validity'] = df.apply(lambda row: 'rejected' if ('rejected' in row['predicted validity1'].strip() or 'rejected' in row['predicted_validity2'].strip()) else 'accepted', axis=1)

#     # Save the updated DataFrame to the same Excel file
#     output_filename = "output_with_predictions1.xlsx"
#     df.to_excel(output_filename, index=False)

#     print("Predictions saved to 'output_with_predictions.xlsx'")






############# storing only the finalvalidity in excel  #############
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
    # Read the Excel file into a DataFrame
    df1 = pd.read_excel('ST-GWC Debit ID 6001367_(DISPUTED CLAIMS) 05152023.xlsx')  # Replace with the actual file path

    # Extract the required columns and convert to datetime objects
    df = df1.iloc[:, :-3]
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

    # Vectorize the comments using the loaded vectorizer
    X_vectorized = vectorizer.transform(df['combined_comments'])

    # Make predictions using the model
    df['predicted_validity2'] = loaded_model.predict(X_vectorized)

    # Add the 'predicted validity1' column with initial value 'Accepted'
    df['predicted validity1'] = 'accepted'

    # Set conditions to update 'predicted validity1' column
    condition1 = (df['Mileage'] > 36000) | (df['time_in_service_years'] > 3) | (df['Vehicle Line AWS'].isin(['M4', 'R2', 'ZE']))
    df.loc[condition1, 'predicted validity1'] = 'rejected'

    # Add the 'final validity' column based on the conditions
    df['final validity'] = df.apply(lambda row: 'rejected' if ('rejected' in row['predicted validity1'].strip() or 'rejected' in row['predicted_validity2'].strip()) else 'accepted', axis=1)

    # Drop unnecessary columns
    df.drop(columns=['combined_comments', 'predicted validity1', 'predicted_validity2'], inplace=True)

    # Save the updated DataFrame to the same Excel file
    output_filename = "output_with_predictions(final).xlsx"
    df.to_excel(output_filename, index=False)

    print("Predictions saved to 'output_with_predictions(final).xlsx'")



############  by taking user input   #################
# import pandas as pd
# import joblib
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# #Load the trained model and vectorizer from the files
# model_filename = "supplier_validity_model.pkl"
# loaded_model = joblib.load(model_filename)

# vectorizer_filename = "count_vectorizer.pkl"
# vectorizer = joblib.load(vectorizer_filename)

# # Load stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Load abbreviations
# abbreviations_df = pd.read_excel('ABBREV.xlsx')
# abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))


# # Function to preprocess text
# def preprocess_text(text):
#     words = nltk.word_tokenize(text.lower())
#     words = [word for word in words if word.isalpha() and word not in stop_words]
#     words = [abbreviations[word] if word in abbreviations else word for word in words]
#     return " ".join(words)

# def get_final_validity(mileage, time_in_service_years, vehicle_line, technician_comments, customer_comments):
#     # Create a DataFrame with the provided input values
#     input_data = pd.DataFrame({
#         'Mileage': [mileage],
#         'time_in_service_years': [time_in_service_years],
#         'Vehicle Line AWS': [vehicle_line],
#         'Technician Comments': [technician_comments],
#         'Customer Comments': [customer_comments]
#     })

#     # Preprocess the comments
#     input_data['Technician Comments'] = input_data['Technician Comments'].apply(preprocess_text)
#     input_data['Customer Comments'] = input_data['Customer Comments'].apply(preprocess_text)

#     # Combine comments
#     input_data['combined_comments'] = input_data['Technician Comments'] + " " + input_data['Customer Comments']

#     # Vectorize the comments using the loaded vectorizer
#     X_vectorized = vectorizer.transform(input_data['combined_comments'])

#     # Make predictions using the model
#     input_data['predicted_validity2'] = loaded_model.predict(X_vectorized)

#     # Add the 'predicted validity1' column with initial value 'Accepted'
#     input_data['predicted validity1'] = 'accepted'

#     # Set conditions to update 'predicted validity1' column
#     condition1 = (input_data['Mileage'] > 36000) | (input_data['time_in_service_years'] > 3) | (input_data['Vehicle Line AWS'].isin(['M4', 'R2', 'ZE']))
#     input_data.loc[condition1, 'predicted validity1'] = 'rejected'

#     # Add the 'final validity' column based on the conditions
#     input_data['final validity'] = input_data.apply(lambda row: 'rejected' if ('rejected' in row['predicted validity1'].strip() or 'rejected' in row['predicted_validity2'].strip()) else 'accepted', axis=1)

#     return input_data['final validity'].values[0]

# if __name__ == '__main__':
#     # Load stopwords
#     nltk.download('stopwords')
#     stop_words = set(stopwords.words('english'))

#     # Load abbreviations
#     abbreviations_df = pd.read_excel('ABBREV.xlsx')
#     abbreviations = dict(zip(abbreviations_df['Acronym'], abbreviations_df['Full Form']))

#     # Load the trained model and vectorizer from the files
#     model_filename = "supplier_validity_model.pkl"
#     loaded_model = joblib.load(model_filename)

#     vectorizer_filename = "count_vectorizer.pkl"
#     vectorizer = joblib.load(vectorizer_filename)

#     # Get user inputs
#     mileage = int(input("Enter the mileage: "))
#     time_in_service_years = float(input("Enter the time in service (in years): "))
#     vehicle_line = input("Enter the vehicle line : ")
#     technician_comments = input("Enter the technician comments: ")
#     customer_comments = input("Enter the customer comments: ")

#     # Get the final validity for the user-entered inputs
#     final_validity = get_final_validity(mileage, time_in_service_years, vehicle_line, technician_comments, customer_comments)
#     print("Final Validity: ", final_validity)
