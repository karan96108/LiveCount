from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time
import numpy as np

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prev_frame_time = 0
    total_faces = 0
    
    try:
        while True:
            happy_count = 0
            not_happy_count = 0
            
            ret, frame = cap.read()
            if not ret:
                break
                
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            total_faces = len(faces)
            
            for (x, y, w, h) in faces:
                try:
                    face = frame[y:y+h, x:x+w]
                    analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    
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
            
            # Display metrics
            cv2.putText(frame, f"People: {total_faces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Happy: {happy_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Not Happy: {not_happy_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
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