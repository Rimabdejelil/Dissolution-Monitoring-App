import argparse
import time
from pathlib import Path
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect(source, model, stride, device='cpu', trace=True, conf_thres=0.25):
    # Initialize logging and select device (CPU or GPU)
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    if half:
        model.half()

    # Load dataset from the source 
    dataset = LoadImages(source, img_size=640, stride=stride)
    
    # Get class names and assign random colors for visualization
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warm-up the model by running a dummy forward pass
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))

    detections = []

    # Loop through the dataset
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)  # Convert image to a tensor and move it to the device
        img = img.half() if half else img.float()  
        img /= 255.0  # Normalize the image to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  
        with torch.no_grad():
            # Perform inference and get predictions
            pred = model(img, augment=False)[0]

        # Apply Non-Maximum Suppression (NMS) to filter predictions
        pred = non_max_suppression(pred, conf_thres, classes=None, agnostic=False)

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # Append the detection to the list
                    bbox = {'coordinates': xyxy, 'class': int(cls), 'confidence': conf.item()}
                    detections.append(bbox)

    return detections
