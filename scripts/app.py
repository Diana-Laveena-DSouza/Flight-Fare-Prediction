from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Models
one_hot_model = pickle.load(open('models/one_hot_model.pkl', 'rb'))
model_regressor = pickle.load(open('models/XGBRegressor.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
# Function for Prediction
def predict():
    input_data = [x for x in request.form.values()]
    data = pd.DataFrame()
    data['Airline'] = pd.Series(input_data[0])
    data['Date_of_Journey'] = pd.Series(input_data[1])
    data['Source'] = pd.Series(input_data[2])
    data['Destination'] = pd.Series(input_data[3])
    data['Route'] = pd.Series(input_data[4])
    data['Dep_Time'] = pd.Series(input_data[5])
    data['Duration'] = pd.Series(input_data[6])
    data['Total_Stops'] = pd.Series(input_data[7])
    data['Additional_Info'] = pd.Series(input_data[8])
    data['Date_of_Journey'] =  pd.to_datetime(data['Date_of_Journey'], format = '%d/%m/%Y')
    splits = []
    for text in data['Duration']:
        if text.find('hr ') > 0:
            splits.append(text.split('hr '))
        else:
            splits.append(text.split('hr'))
    Duration = []
    for i in range(len(splits)):
        if splits[i][0].find('min') > 0:
            Duration.append(int(splits[i][0][:-3]))
        elif splits[i][1][:-1] != '':
            Duration.append((int(splits[i][0]) * 60) + int(splits[i][1][:-3]))
        else:
            Duration.append((int(splits[i][0]) * 60))
    hr = []
    min = []
    for text in data['Dep_Time']:
        hr.append(text.split(':')[0])
        min.append(text.split(':')[1])
    data['Duration'] = Duration
    data['Dep_Time_hr'] = np.array(hr).astype('int')
    data['Dep_Time_min'] = np.array(min).astype('int')
    data_cat = one_hot_model.transform(data[['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info']]).toarray()
    data_cat_final = pd.DataFrame(data_cat)
    data_cat_final.columns = one_hot_model.get_feature_names_out(['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info'])
    data_final = pd.concat([data, data_cat_final], axis=1)
    data_final['Month'] = data_final['Date_of_Journey'].dt.month
    data_final['Date'] = data_final['Date_of_Journey'].dt.day
    Features = data_final.drop(['Dep_Time', 'Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info'], axis=1)
    prediction = model_regressor.predict(Features)
    return render_template('index.html', prediction_text = 'Rs. ' + str(int(prediction[0])))


if __name__ == "__main__":
    app.run(debug = True)
