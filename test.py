from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time
import numpy as np

app = Flask(__name__)

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def gen_frames():
    try:
        # Try multiple camera indices
        for camera_index in range(2):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                break
        
        if not cap.isOpened():
            raise Exception("Could not open camera")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        prev_frame_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        person_count = len(faces)
        happy_count = 0
        not_happy_count = 0

        for (x, y, w, h) in faces:
            try:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = preprocess_face(face_roi)
                
                analysis = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, confidence = dominant_emotion

                if emotion_name in ['happy', 'surprise'] and confidence > 50:
                    happy_count += 1
                    emotion_text = f"Happy ({int(confidence)}%)"
                    color = (0, 255, 0)
                else:
                    not_happy_count += 1
                    emotion_text = f"Not Happy ({int(confidence)}%)"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Error analyzing face: {str(e)}")
                continue

        cv2.putText(frame, f"Total People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Happy: {happy_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Not Happy: {not_happy_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Camera error: {str(e)}")
        # Create error frame
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, "Camera access error", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if 'cap' in locals():
            cap.release()    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)