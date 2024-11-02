
from flask import Flask, Response
import cv2
import pypylon.pylon as py
from datetime import datetime
from gevent.pywsgi import WSGIServer
app = Flask(__name__)


icam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
icam.Open()
icam.ChunkModeActive.SetValue(True)



icam.ChunkSelector.SetValue("ExposureTime")
icam.ChunkEnable.SetValue(True)
icam.ExposureTime = 80000
icam.PixelFormat = "BGR8"

@app.route('/')
def index():
    return "Default Message"

def gen():
    while True:
      
        image = icam.GrabOne(4000) 
        image = image.Array
        image = cv2.resize(image, (0,0), fx=0.8366, fy=1, interpolation=cv2.INTER_LINEAR)
       
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,str(datetime.now()),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', image)        
        frame = jpeg.tobytes()        
     
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n'
               b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
               b'\r\n' + frame + b'\r\n')
@app.route('/video_feed') 
def video_feed():


    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':

    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
