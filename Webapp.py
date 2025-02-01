import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dataset = pd.read_csv('animal_disease_dataset.csv')

onehotencoder = OneHotEncoder(sparse_output=False)
minmaxscaler = MinMaxScaler()

#scaling numerical i.e. age and temperature columns
scale_columns = ['Age', 'Temperature']
dataset1 = dataset.copy()

dataset1[scale_columns] = minmaxscaler.fit_transform(dataset1[scale_columns])

#Splitting the dataset into train and test sets
dataset2 = dataset1.drop('Disease', axis=1)
target = dataset1['Disease']

train_data, test_data, train_target, test_target = train_test_split(dataset2, target, test_size=0.3, random_state=42)

#One hot encoding of categorical columns in the dataset
columns_category = train_data.select_dtypes('object').columns.tolist()

dataset3 = dataset1[columns_category]
onehotencoder.fit(dataset3)

#print all the encoded columns
columns_encoded = list(onehotencoder.get_feature_names_out(columns_category))

train_data[columns_encoded] = onehotencoder.transform(train_data[columns_category])
test_data[columns_encoded] = onehotencoder.transform(test_data[columns_category])

train_data.drop(columns=['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3'], inplace=True)
test_data.drop(columns=['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3'], inplace=True)

X_train = train_data
X_test = test_data
y_train = train_target
y_test = test_target

#Building the model using random forest classifier
randomforest = RandomForestClassifier(random_state= 42)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100

#Metrics of Random Forest Classifier
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def main():
    return render_template('home.html')

@app.route('/Predict', methods=['POST'])
def home():
    animalname = request.form['animalname']
    age = request.form['age']
    temperature = request.form['temperature']
    first_symptom = request.form['first_symptom']
    second_symptom = request.form['second_symptom']
    third_symptom = request.form['third_symptom']
    
    #formatting the user input data
    user_input = {
        'Animal': animalname,
        'Age': age,
        'Temperature': temperature,
        'Symptom 1': first_symptom,
        'Symptom 2': second_symptom,
        'Symptom 3': third_symptom
    }
    #predicting the disease for the user input data
    columns_numerical = ['Age', 'Temperature']
    columns_categorical = ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']
    input_data = pd.DataFrame(user_input, index=[0])
    input_data[columns_numerical] = minmaxscaler.transform(input_data[columns_numerical])
    input_data[columns_encoded] = onehotencoder.transform(input_data[columns_categorical])
    input_data.drop(columns=columns_categorical, inplace=True)
    
    name=[]
    prediction = randomforest.predict(input_data)[0]
    probability = randomforest.predict_proba(input_data)[0][list(randomforest.classes_).index(prediction)]
    probability = round((probability *100 ),2)
    name.append(prediction)
    name.append(probability)
    return render_template('action.html',data=name, accuracy=accuracy, precision=precision, recall=recall, f1=f1, conf_matrix=conf_matrix)

if __name__ == "__main__":
    app.run(debug=True)