import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Create an object of the class Flask
app = Flask(__name__)

# load trained model
model = pickle.load(open('model.pkl','rb'))

# load preprocessing objects needed 
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# url/
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # get form data
    year = int(request.form.get('year'))
    engine_size = float(request.form.get('engine_size'))
    cylinders = int(request.form.get('cylinders'))
    transmission = request.form.get('transmission')
    fuel = request.form.get('fuel')
    coemissions = float(request.form.get('coemissions'))
    make = request.form.get('make')
    model_name = request.form.get('model')
    vehicle_class = request.form.get('vehicle_class')
    
    # coverting into dataframe as we had originally for training 
    df = pd.DataFrame({
        'MAKE': [make],
        'MODEL': [model_name],
        'VEHICLE CLASS': [vehicle_class],
        'ENGINE SIZE': [engine_size],
        'CYLINDERS': [cylinders],
        'TRANSMISSION': [transmission],
        'FUEL': [fuel],
        'COEMISSIONS': [coemissions]
    })

    numerical_features = df.select_dtypes(include=['int64','float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    df_num_scaled = pd.DataFrame(scaler.transform(df[numerical_features]), columns=numerical_features, index=df.index)

    one_hot_columns = ['VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
    df_encoded = pd.DataFrame(encoder.transform(df[one_hot_columns]).toarray(), index=df.index, columns=encoder.get_feature_names_out(one_hot_columns))

    df['MAKE_freq'] = df['MAKE'].map(df['MAKE'].value_counts(normalize=True))
    df['MODEL_freq'] = df['MODEL'].map(df['MODEL'].value_counts(normalize=True))
    df['MODEL_freq'] = df['MODEL_freq'].fillna(0)

    df = df.drop(columns=categorical_features)
    df = pd.concat([df, df_encoded], axis=1)
    df_scaled = df.drop(columns=numerical_features)
    df_scaled = pd.concat([df, df_num_scaled], axis=1)

    # predicting with model
    prediction = model.predict(df)
    output = round(prediction[0],2)

    return render_template('index.html', prediction_text=f"Predicted Fuel Consumption: {output}")

if __name__=='__main__':
    app.run(debug=True)



    # # applying one hot encoding for categorical variables using saved encoder
    # encoder_columns = [transmission,fuel,vehicle_class]
    # encoder_columns_reshaped = np.array(encoder_columns).reshape(1,-1)
    # encoded = encoder.transform(encoder_columns_reshaped)

    # # encoding 'make' and 'model_name' using previously loaded objects
    # make_encoded = make_freq.get(make, 0)
    # model_encoded = model_freq.get(model_name, 0)

    # # create dataframe for prediction

    # input_data = pd.DataFrame({
    #     'Year':[year],
    #     'ENGINE SIZE': [engine_size],
    #     'CYLINDERS': [cylinders],
    #     'TRANSMISSION': [encoded],
    #     'FUEL': [encoded],
    #     'COEMISSIONS': [coemissions],
    #     'MAKE_freq': [make_encoded],
    #     'MODEL_freq': [model_encoded],
    #     'VEHICLE CLASS': [encoded]
    # })

    # # scaling the data 
    # input_data_scaled = scaler.transform(input_data)
