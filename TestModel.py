import numpy as np
import joblib  # Required to load MinMaxScaler
from keras.models import load_model

feature_scaler = joblib.load('feature_scaler.joblib')

target_scaler = joblib.load('target_scaler.joblib')

model = load_model('lstm_model.h5')

new_time_area = np.array([[1348   ,17876.0]])

scaled_new_data = feature_scaler.transform(new_time_area)

previous_data_points = np.array([
    [1346,18040.0],
    [1347,17876.0]
])

scaled_previous_data_points = feature_scaler.transform(previous_data_points)

sequence = np.concatenate((scaled_previous_data_points, scaled_new_data), axis=0)


window_size =3

X_new = []
for i in range(len(sequence) - window_size + 1):
    X_new.append(sequence[i:i+window_size])

X_new = np.array(X_new)

predictions = model.predict(X_new)

predictions_original_scale = target_scaler.inverse_transform(predictions)

print("Predictions (Dissolution Rate):")
for pred in predictions_original_scale:
    print(f"Dissolution Rate: {pred[0]}")