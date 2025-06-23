from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Cargar modelo YOLO entrenado para detectar botellas
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)
            # Filtrar solo objetos etiquetados como "bottle"
            bottles = results.pandas().xyxy[0]
            bottles = bottles[bottles['name'] == 'bottle']

            for _, row in bottles.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
