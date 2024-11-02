import sys
import subprocess
import os
import cv2
import time
from pypylon import pylon

tl_factory = pylon.TlFactory.GetInstance()

camera = pylon.InstantCamera()

camera.Attach(tl_factory.CreateFirstDevice())

try:
    camera.Open()

    camera.PixelFormat = "BGR8"

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    image_count = 657

    dataset_dir = 'dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'dissolved'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'non_dissolved'), exist_ok=True)

    while True:
        grab_result = camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            img = grab_result.Array

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            filename = os.path.join(dataset_dir, f"{image_count}.jpg")

            cv2.imwrite(filename, img)
            print(f"Image saved as: {filename}")

            python_executable = sys.executable
            process = subprocess.Popen([
                python_executable, "detect.py", "--img-size", "640", "--source", filename, "--conf", "0.8", "--weights", "YoloMoodel.pt"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
            )

           

            image_count += 1
            grab_result.Release()

        time.sleep(5)

except KeyboardInterrupt:
    print("Image capture stopped by user.")

finally:
    camera.StopGrabbing()

    camera.Close()
