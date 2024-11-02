import cv2
import pypylon.pylon as py
import time
import numpy as np
from collections import deque
import joblib
from keras.models import load_model
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import TracedModel
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pathlib import Path
from utils.plots import plot_one_box
import base64
import threading
from detection import detect

app = Flask(__name__)
app.config['SECRET_KEY'] = "your_secret_key"
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Basler camera setup
icam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
icam.Open()
icam.PixelFormat = "BGR8"
icam.ChunkSelector.SetValue("ExposureTime")
icam.ChunkEnable.SetValue(True)
icam.ExposureTime = 80000

# Initialize variables
initial_area = None
previous_data_points = deque(maxlen=2)
start_time = time.time()
stop_streaming = False
global_state = {
    'cls': None,
    'image_base64': None
}

# Load YOLO model
model = attempt_load(weights='YoloMoodel.pt', map_location='cpu')
stride = int(model.stride.max())
img_size = check_img_size(img_size=640, s=stride)
model = TracedModel(model, device='cpu', img_size=640)

# Load scalers and LSTM models
def load_scalers_and_model(model_path):
    feature_scaler = joblib.load(Path(model_path) / 'feature_scaler.joblib')
    target_scaler = joblib.load(Path(model_path) / 'target_scaler.joblib')
    lstm_model = load_model(Path(model_path) / 'lstm_model.h5')
    return feature_scaler, target_scaler, lstm_model

feature_scaler_1, target_scaler_1, lstm_model_1 = load_scalers_and_model('One_solid_lstm')
feature_scaler_2, target_scaler_2, lstm_model_2 = load_scalers_and_model('Solid_into_pieces_lstm')

# Function to predict dissolution rate
def predict_dissolution_rate(previous_data_points, new_data_point, feature_scaler, target_scaler, lstm_model):
    sequence = np.concatenate((previous_data_points, [new_data_point]), axis=0)
    scaled_sequence = feature_scaler.transform(sequence)
    X = np.expand_dims(scaled_sequence, axis=0)
    prediction = lstm_model.predict(X)
    dissolution_rate = target_scaler.inverse_transform(prediction)[0][0]
    return dissolution_rate

def emit_cls_update():
    global global_state
    socketio.emit('cls_update', global_state)

def background_worker():
    global initial_area, previous_data_points, start_time, global_state

    while True:
        # Grab image
        image = icam.GrabOne(4000).Array
        image_path = "current_frame.jpg"
        cv2.imwrite(image_path, image)

        # Call detect function
        detections = detect(image_path, model, stride)
        elapsed_time = time.time() - start_time

        cls = None

        for detection in detections:
            xyxy = detection['coordinates']
            cls = detection['class']
            conf = detection['confidence']

            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)
            if initial_area is None:
                initial_area = area

            # Calculate dissolved area percentage
            dissolved_percentage = ((initial_area - area) / initial_area) * 100
            dissolved_percentage_label = f'Dissolved Area: {dissolved_percentage:.2f}%'

            # Feed input into model
            if len(previous_data_points) == 2:
                timestamp = elapsed_time
                if cls == 1:
                    dissolution_rate = predict_dissolution_rate(list(previous_data_points), (timestamp, area), feature_scaler_1, target_scaler_1, lstm_model_1)
                elif cls == 2:
                    dissolution_rate = predict_dissolution_rate(list(previous_data_points), (timestamp, area), feature_scaler_2, target_scaler_2, lstm_model_2)
                else:
                    dissolution_rate = 0

                if dissolution_rate > 0:
                    remaining_time = area / dissolution_rate
                    estimated_time = int(remaining_time // 60)
                    # Draw Box around detected object & display remaining time
                    remaining_time_label = f'Remaining Time: {estimated_time:.2f} minutes'
                    plot_one_box((0, 300, 0, 0), image, color=(10, 10, 10), labels=[remaining_time_label], line_thickness=8)

                if cls == 0 :
                    dissolved_percentage= 100


                dissolved_percentage_label = f'Dissolved Area: {dissolved_percentage:.2f}%'

            previous_data_points.append((elapsed_time, area))
            custom_names = ["dissolved", "one_solid", "solid_into_pieces"]
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in custom_names]
            labels = [f'{custom_names[int(cls)]} {conf:.2f}']
            plot_one_box(xyxy, image, color=colors[int(cls)], labels=labels, line_thickness=10)

            cv2.putText(image, dissolved_percentage_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', image)
        global_state['cls'] = cls
        global_state['image_base64'] = base64.b64encode(jpeg.tobytes()).decode('utf-8')

def background_emit():
    while True :
        socketio.emit('cls_update', global_state)
        socketio.sleep(1)

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(background_emit)
 

@socketio.on('detection_started')
def start_threading():
    thread = threading.Thread(target=background_worker)
    thread.daemon = True
    thread.start()

#@socketio.on('cls_update')
#def emit_cls_update():
  #  global global_state
   # socketio.emit('cls_update', global_state)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5080)
