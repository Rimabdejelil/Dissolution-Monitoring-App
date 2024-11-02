import argparse
import time
from pathlib import Path
import csv
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
from keras.models import load_model


from collections import deque

def load_scalers_and_model(model_path):
    feature_scaler = joblib.load(Path(model_path) / 'feature_scaler.joblib')
    target_scaler = joblib.load(Path(model_path) / 'target_scaler.joblib')
    lstm_model = load_model(Path(model_path) / 'lstm_model.h5')
    return feature_scaler, target_scaler, lstm_model


def predict_dissolution_rate(previous_data_points, new_data_point, feature_scaler, target_scaler, lstm_model):
    sequence = np.concatenate((previous_data_points, [new_data_point]), axis=0)
    scaled_sequence = feature_scaler.transform(sequence)
    X = np.expand_dims(scaled_sequence, axis=0)
    prediction = lstm_model.predict(X)
    dissolution_rate = target_scaler.inverse_transform(prediction)[0][0]
    return dissolution_rate


start_time = time.time()

def detect(save_img=False):

    detection_status = False

    initial_area = None
    fps = 30
    estimated_time = 0
    remaining_time = 0

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt') 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  


    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  

    model = attempt_load(weights, map_location=device)  
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride)  

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    last_save = time.time() 
    csv_file = open('Data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "x1", "y1", "x2", "y2"])  

    feature_scaler_1, target_scaler_1, lstm_model_1 = load_scalers_and_model('One_solid_lstm')
    feature_scaler_2, target_scaler_2, lstm_model_2 = load_scalers_and_model('Solid_into_pieces_lstm')


    elapsed_time = 0.0
    previous_data_points = deque(maxlen=2)

    for path, img, im0s, vid_cap in dataset:
        elapsed_time = time.time() - start_time

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():  
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        if not webcam and vid_cap:
            fps = vid_cap.get(cv2.CAP_PROP_FPS) 

        for i, det in enumerate(pred):  

            if webcam:  
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name)  
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):

                        
                    timestamp = elapsed_time
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])


                    if initial_area is None :
                        initial_area=area

                
                    if len(previous_data_points) == 2:

                            
                        
                        percentage_dissolved = 100 - ((area/initial_area)*100)

                        if cls == 0 : 
                            percentage_dissolved = 100

                        percentage_text = f'Dissolved: {percentage_dissolved:.2f}%'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 2.0  
                        font_thickness = 4  
                        text_size, _ = cv2.getTextSize(percentage_text, font, font_scale, font_thickness)
                        text_width, text_height = text_size                              

                        text_position_x = im0.shape[1] - text_width - 30  
                        text_position_y = im0.shape[0] - 50  
                        cv2.putText(im0, percentage_text, (text_position_x, text_position_y), font, font_scale, (255, 255, 255), font_thickness)


                        if cls == 1:
                            dissolution_rate = predict_dissolution_rate(list(previous_data_points), (timestamp, area), feature_scaler_1, target_scaler_1, lstm_model_1)
                        elif cls == 2:
                            dissolution_rate = predict_dissolution_rate(list(previous_data_points), (timestamp, area), feature_scaler_2, target_scaler_2, lstm_model_2)
                        else:
                            dissolution_rate = 0  

                        if dissolution_rate > 0:
                            remaining_time = area / dissolution_rate
                            estimated_time = int(remaining_time//60)
                            print(dissolution_rate)
                            print(area)
                            print(timestamp)
                            print(cls)

                        else:
                            remaining_time = float('inf')

                    
                    if cls == 1 : 
                            
                        print("Non Dissolved detected.")
                        if detection_status is not None:
                            detection_status =False
                            with open('dissolution_status.txt', 'w') as f:
                                f.write(str(detection_status))

                    if cls == 0 :
                        print("Dissolved detected.")
                        remaining_time = 0
                        estimated_time = 0
                        percentage_dissolved = 100
                        if detection_status is not None:
                            detection_status = True
                            with open('dissolution_status.txt', 'w') as f:
                                f.write(str(detection_status))

                      
                    
                previous_data_points.append((timestamp, area))
                print(previous_data_points)
                elapsed_time += 1.0 / fps  

                if save_txt:  
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh) 
                    with open(txt_path + '.txt', 'a') as f:
                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  
                    remaining_time_label = f'Remaining Time: {estimated_time:.2f} minutes'
                    plot_one_box((0,300,0,0), im0, color=(10,10,10), labels=[remaining_time_label], line_thickness=8)

        
                    labels = [f'{names[int(cls)]} {conf:.2f}']
                    plot_one_box(xyxy, im0, color=colors[int(cls)], labels=labels, line_thickness=10)


            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if view_img:
                im0_resized = cv2.resize(im0, (800, 600))
                cv2.imshow(str(p), im0_resized)
                cv2.waitKey(1)

            if save_img:

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        if vid_cap:  
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            last_save = time.time()
                        
            xyxy = [float(i) for i in xyxy]
            try:
                csv_writer.writerow([timestamp, *xyxy])
                csv_file.flush()  
            except Exception as e:
                print(f"Error writing to CSV: {e}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
