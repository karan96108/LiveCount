from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time
import numpy as np
import warnings
import speech_recognition as sr
import threading
import wave
import statistics
import math
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
face_emotion = "Unknown"
voice_emotion = "Not detected"
voice_analysis_done = False
voice_analysis_running = False

def detect_face_emotion(frame):
    """Detect emotion from face using DeepFace"""
    global face_emotion
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name, confidence = dominant_emotion
        
        face_emotion = f"{emotion_name.capitalize()} ({int(confidence)}%)"
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        face_emotion = "Unknown"

def analyze_voice_emotion():
    """Analyze voice emotion using speech_recognition"""
    global voice_emotion, voice_analysis_done, voice_analysis_running
    
    voice_analysis_running = True
    recognizer = sr.Recognizer()
    try:
        # Use the default microphone as the audio source
        with sr.Microphone() as source:
            print("Please speak for 5 seconds to analyze your voice emotion...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Record for 5 seconds
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            try:
                # Try to recognize the speech
                text = recognizer.recognize_google(audio_data)
                print(f"Recognized: {text}")
                
                # Basic emotion analysis based on text and audio characteristics
                audio_data_bytes = audio_data.get_raw_data()
                
                # Convert to numpy array of int16
                int_data = np.frombuffer(audio_data_bytes, dtype=np.int16)
                
                # Calculate basic audio features
                energy = np.mean(np.abs(int_data))
                zero_crossings = np.sum(np.abs(np.diff(np.signbit(int_data))))
                
                # Simple rule-based emotion detection
                if energy > 10000 and zero_crossings > len(int_data) * 0.1:
                    voice_emotion = "Excited/Angry"
                elif energy > 7000:
                    voice_emotion = "Happy"
                elif zero_crossings < len(int_data) * 0.05:
                    voice_emotion = "Sad"
                else:
                    voice_emotion = "Neutral"
                
                print(f"Voice emotion detected: {voice_emotion}")
                print(f"Energy: {energy}, Zero crossings: {zero_crossings}")
                
            except sr.UnknownValueError:
                voice_emotion = "Speech not understood"
                print("Could not understand audio")
            except sr.RequestError as e:
                voice_emotion = "Service error"
                print(f"Could not request results; {e}")
            except Exception as e:
                voice_emotion = "Analysis error"
                print(f"Error analyzing voice: {e}")
            
    except Exception as e:
        voice_emotion = "Capture error"
        print(f"Error capturing voice: {e}")
    
    voice_analysis_done = True
    voice_analysis_running = False

def start_voice_analysis():
    """Start voice analysis in a separate thread"""
    voice_thread = threading.Thread(target=analyze_voice_emotion)
    voice_thread.daemon = True
    voice_thread.start()
    return voice_thread

def gen_frames():
    """Generate video frames with emotion detection"""
    global voice_emotion, voice_analysis_done, voice_analysis_running
    
    # Start voice analysis thread at the beginning
    if not voice_analysis_done and not voice_analysis_running:
        start_voice_analysis()
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prev_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        new_frame_time = time.time()
        fps_float = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_display = int(fps_float) # Ensure FPS is an integer for display

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )
        num_people = len(faces) # Get the number of detected people
        
        for (x, y, w, h) in faces:
            try:
                face_roi = frame[y:y+h, x:x+w]
                detect_face_emotion(face_roi)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, face_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            except Exception as e:
                print(f"Error analyzing face: {str(e)}")
                continue
        
        cv2.putText(frame, f'FPS: {fps_display}', (10, 50), # Display integer FPS
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Face Emotion: {face_emotion}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Voice Emotion: {voice_emotion}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        if num_people > 1: # Display number of people if more than 1
            cv2.putText(frame, f'Number of People: {num_people}', (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show voice analysis status
        if voice_analysis_running:
            cv2.putText(frame, "Voice analysis in progress...", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Cleanup
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_voice')
def analyze_voice_route():
    """Route to trigger voice analysis"""
    global voice_analysis_done, voice_analysis_running
    if not voice_analysis_running:
        voice_analysis_done = False
        start_voice_analysis()
        return "Voice analysis started"
    else:
        return "Voice analysis already in progress"

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        pass  # Cleanup if needed