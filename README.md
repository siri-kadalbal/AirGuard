# AirGuard
An AI-powered system designed to predict the AQI values based on key environmental factors, such as pollutant concentrations and geographic data. 

This project aims to predict the Air Quality Index (AQI) for different locations based on various environmental parameters such as PM2.5, NO2, O3, Latitude, Longitude, Arithmetic Mean, and other features. We use a Neural Network model built with TensorFlow/Keras to perform this regression task, which can be useful for real-time air quality monitoring.

Project Features
Dataset: Utilizes air quality data for 2024 (daily_88101_2024.csv), including pollutants like PM2.5, NO2, and O3.

Target Variable: AQI (Air Quality Index)

Features: Arithmetic Mean, Day of Year, Latitude, Longitude, 1st Max Value.

Model Type: Neural Network (using Keras and TensorFlow).

Evaluation: Mean Absolute Error (MAE) used to evaluate model performance.

Requirements
To run this project, you will need the following Python packages:

tensorflow

pandas

numpy

scikit-learn

matplotlib (for visualizations)

pydot and graphviz (for model visualization)

You can install the required packages using pip:

bash
Copy
Edit
pip install tensorflow pandas numpy scikit-learn matplotlib pydot graphviz
How It Works
Data Loading: The dataset (daily_88101_2024.csv) is loaded into a pandas DataFrame.

Data Preprocessing: Missing values are dropped, and we filter the dataset to only include rows with relevant pollutants (PM2.5, NO2, O3).

Feature Engineering:

The Date Local column is converted to datetime format, and the Day of Year feature is extracted.

We select relevant columns for the feature matrix X and target variable y.

Model Training:

The data is split into training and testing sets.

Features are standardized using StandardScaler.

A neural network model is defined with Dense layers.

Model Evaluation: The model is trained for 50 epochs, and the Mean Absolute Error (MAE) is calculated on the test set to evaluate the model's performance.

Model Visualization: The architecture of the neural network is visualized and saved as an image.

Neural Network Architecture
The model consists of the following layers:

Input Layer: Takes the features: Arithmetic Mean, Day of Year, Latitude, Longitude, 1st Max Value.

Hidden Layers:

Dense layer with 64 units and ReLU activation

Dense layer with 32 units and ReLU activation

Dense layer with 16 units and ReLU activation

Output Layer: A single neuron outputting the predicted AQI value.

Example Usage
python
Copy
Edit
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
df = pd.read_csv("daily_88101_2024.csv")
df.dropna(inplace=True)
df_pollutants = df[df['Parameter Name'].str.contains('PM2.5|NO2|O3', case=False, na=False)]
df_pollutants['Date Local'] = pd.to_datetime(df_pollutants['Date Local'])
df_pollutants['Day of Year'] = df_pollutants['Date Local'].dt.dayofyear

X = df_pollutants[['Arithmetic Mean', 'Day of Year', 'Latitude', 'Longitude', '1st Max Value']]
y = df_pollutants['AQI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.2f}")

Model Evaluation
The final model is evaluated using Mean Absolute Error (MAE), which measures the average magnitude of errors in the predictions. A lower MAE indicates better model performance.

Visualizing the Neural Network
After defining the model, you can visualize its architecture:

python
Copy
Edit
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
This will generate a graphical representation of the neural network architecture.

Conclusion
This project provides a basic framework for predicting air quality based on multiple environmental features. The neural network model can be further improved by incorporating additional features, experimenting with more complex architectures, or using different evaluation metrics. This tool can be useful for monitoring air quality in real-time applications.


