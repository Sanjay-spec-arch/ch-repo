# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the training data (used only for column reference)
df_1 = pd.read_csv("first_telc.csv")

# Group the tenure column in the training set
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df_1['tenure_group'] = pd.cut(df_1.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
df_1.drop(columns=['tenure'], axis=1, inplace=True)

# Create dummy variables from the training data to get full column set
train_df_dummies = pd.get_dummies(df_1[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])
dummy_columns = train_df_dummies.columns  # Save columns for later reindexing

# Load trained model
model = pickle.load(open("C:\Users\nikla\Desktop\ch pull\MLProject-ChurnPrediction\model.sav","rb"))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the training data (used only for column reference)
df_1 = pd.read_csv("first_telc.csv")

# Group the tenure column in the training set
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df_1['tenure_group'] = pd.cut(df_1.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
df_1.drop(columns=['tenure'], axis=1, inplace=True)

# Create dummy variables from the training data to get full column set
train_df_dummies = pd.get_dummies(df_1[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])
dummy_columns = train_df_dummies.columns  # Save columns for later reindexing

# Load trained model
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    input_data = [
        request.form['query1'],  # SeniorCitizen
        request.form['query2'],  # MonthlyCharges
        request.form['query3'],  # TotalCharges
        request.form['query4'],  # gender
        request.form['query5'],  # Partner
        request.form['query6'],  # Dependents
        request.form['query7'],  # PhoneService
        request.form['query8'],  # MultipleLines
        request.form['query9'],  # InternetService
        request.form['query10'], # OnlineSecurity
        request.form['query11'], # OnlineBackup
        request.form['query12'], # DeviceProtection
        request.form['query13'], # TechSupport
        request.form['query14'], # StreamingTV
        request.form['query15'], # StreamingMovies
        request.form['query16'], # Contract
        request.form['query17'], # PaperlessBilling
        request.form['query18'], # PaymentMethod
        request.form['query19']  # tenure
    ]

    # Create DataFrame
    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Preprocess new_df
    new_df['tenure_group'] = pd.cut(new_df['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    # Create dummies and reindex to match training columns
    new_df_dummies = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])

    # Match the columns with training dummy columns
    new_df_dummies = new_df_dummies.reindex(columns=dummy_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(new_df_dummies)[0]
    probability = model.predict_proba(new_df_dummies)[0][1]

    if prediction == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"
    
    o2 = "Confidence: {:.2f}%".format(probability * 100)

    return render_template('home.html', 
                           output1=o1, 
                           output2=o2,
                           query1=request.form['query1'],
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'],
                           query6=request.form['query6'],
                           query7=request.form['query7'],
                           query8=request.form['query8'],
                           query9=request.form['query9'],
                           query10=request.form['query10'],
                           query11=request.form['query11'],
                           query12=request.form['query12'],
                           query13=request.form['query13'],
                           query14=request.form['query14'],
                           query15=request.form['query15'],
                           query16=request.form['query16'],
                           query17=request.form['query17'],
                           query18=request.form['query18'],
                           query19=request.form['query19'])

if __name__ == "__main__":
    app.run(debug=True)
", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    input_data = [
        request.form['query1'],  # SeniorCitizen
        request.form['query2'],  # MonthlyCharges
        request.form['query3'],  # TotalCharges
        request.form['query4'],  # gender
        request.form['query5'],  # Partner
        request.form['query6'],  # Dependents
        request.form['query7'],  # PhoneService
        request.form['query8'],  # MultipleLines
        request.form['query9'],  # InternetService
        request.form['query10'], # OnlineSecurity
        request.form['query11'], # OnlineBackup
        request.form['query12'], # DeviceProtection
        request.form['query13'], # TechSupport
        request.form['query14'], # StreamingTV
        request.form['query15'], # StreamingMovies
        request.form['query16'], # Contract
        request.form['query17'], # PaperlessBilling
        request.form['query18'], # PaymentMethod
        request.form['query19']  # tenure
    ]

    # Create DataFrame
    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Preprocess new_df
    new_df['tenure_group'] = pd.cut(new_df['tenure'].astype(int), range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    # Create dummies and reindex to match training columns
    new_df_dummies = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']])

    # Match the columns with training dummy columns
    new_df_dummies = new_df_dummies.reindex(columns=dummy_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(new_df_dummies)[0]
    probability = model.predict_proba(new_df_dummies)[0][1]

    if prediction == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"
    
    o2 = "Confidence: {:.2f}%".format(probability * 100)

    return render_template('home.html', 
                           output1=o1, 
                           output2=o2,
                           query1=request.form['query1'],
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'],
                           query6=request.form['query6'],
                           query7=request.form['query7'],
                           query8=request.form['query8'],
                           query9=request.form['query9'],
                           query10=request.form['query10'],
                           query11=request.form['query11'],
                           query12=request.form['query12'],
                           query13=request.form['query13'],
                           query14=request.form['query14'],
                           query15=request.form['query15'],
                           query16=request.form['query16'],
                           query17=request.form['query17'],
                           query18=request.form['query18'],
                           query19=request.form['query19'])

if __name__ == "__main__":
    app.run(debug=True)
