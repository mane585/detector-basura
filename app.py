
from flask import Flask, render_template, Response, request
import cv2
import torch
import requests

app = Flask(__name__)
ESP32_URL = "http://192.168.1.100"  # Cambia esto por la IP de tu ESP32

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
cap = cv2.VideoCapture(0)  # Usa cámara externa

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            for *box, conf, cls in results.xyxy[0]:
                if results.names[int(cls)] == "bottle":
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Bottle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move', methods=['POST'])
def move():
    direction = request.form['direction']
    try:
        requests.get(f"{ESP32_URL}/{direction}")
    except:
        pass
    return ('', 204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
