import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time

# Page config
st.set_page_config(page_title="People Counter", layout="wide")

# Initialize model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

# Audio file mapping
AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3",
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.last_audio_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=5)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process every 3rd frame for performance
        if self.frame_count % 3 == 0:
            # Run YOLO detection
            results = model(img, 
                          verbose=False, 
                          imgsz=640,
                          conf=0.4,
                          device='cpu')
            
            person_boxes = []
            count = 0
            
            # Filter for person class (class 0)
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
                    count += 1
            
            count = min(count, 5)
            self.detection_history.append(count)
            
            # Smooth detection with moving average
            if len(self.detection_history) > 0:
                avg_count = int(np.median(self.detection_history))
                self.person_count = avg_count
            
            # Draw bounding boxes
            for (x1, y1, x2, y2, conf) in person_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Person {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw count overlay
        text = f"People Detected: {self.person_count}"
        font_scale = 1.2
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (text_width + 30, text_height + 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        cv2.putText(img, text, (20, 50),
                    font, font_scale, (0, 255, 0), thickness)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🎥 Real-Time People Detection")
st.markdown("### Detects people and plays audio based on count")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Live Camera Feed")
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="people-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PersonDetector,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.markdown("#### Status")
    
    if ctx.video_processor:
        count_placeholder = st.empty()
        audio_placeholder = st.empty()
        
        # Update display
        while ctx.state.playing:
            if hasattr(ctx.video_processor, 'person_count'):
                current_count = ctx.video_processor.person_count
                count_placeholder.metric("People Count", current_count)
                
                # Play audio when count changes
                if current_count != ctx.video_processor.last_audio_count and current_count > 0:
                    if current_count in AUDIO_FILES:
                        audio_file = AUDIO_FILES[current_count]
                        try:
                            audio_placeholder.audio(audio_file, autoplay=True)
                            ctx.video_processor.last_audio_count = current_count
                        except:
                            audio_placeholder.warning(f"Audio file {audio_file} not found")
                elif current_count == 0:
                    ctx.video_processor.last_audio_count = 0
                    audio_placeholder.empty()
            
            time.sleep(0.5)
    else:
        st.info("👆 Click 'START' to begin detection")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. Click **START** to activate your camera
    2. Allow camera permissions in your browser
    3. The app will detect people and play corresponding audio
    4. Make sure your MP3 files are in the repo root
    
    **Required Files:**
    - `1_person.mp3`
    - `2_people.mp3`
    - `3_people.mp3`
    - `4_people.mp3`
    - `5_people.mp3`
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 and Streamlit*")
