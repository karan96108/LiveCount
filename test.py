import streamlit as st
import cv2
from deepface import DeepFace
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# Configure TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def main():
    try:
        st.title("Live Emotion Detection")
        
        # Camera selection
        camera_options = ["Default Camera (0)", "External Camera (1)", "Virtual Camera (2)"]
        selected_camera = st.selectbox("Select Camera", range(len(camera_options)), format_func=lambda x: camera_options[x])
        
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Create placeholders
        frame_placeholder = st.empty()
        col1, col2, col3, col4 = st.columns(4)
        person_count_text = col1.empty()
        happy_count_text = col2.empty()
        not_happy_count_text = col3.empty()
        fps_text = col4.empty()
        
        # Start video capture with retry
        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            st.error(f"Failed to open camera {selected_camera}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.putText(frame, "No Camera Available", (50, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame_placeholder.image(frame, channels="RGB")
            return
            
        prev_frame_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Detect faces
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
            
            # Process each face
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
                    cv2.putText(frame, emotion_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                except Exception as e:
                    st.error(f"Error analyzing face: {str(e)}")
                    continue
            
            # Update metrics
            person_count_text.metric("Total People", person_count)
            happy_count_text.metric("Happy", happy_count)
            not_happy_count_text.metric("Not Happy", not_happy_count)
            fps_text.metric("FPS", int(fps))
            
            # Convert BGR to RGB for streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update frame
            frame_placeholder.image(frame, channels="RGB")
            
            # Check for stop button
            if st.button('Stop'):
                break
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == '__main__':
    main()