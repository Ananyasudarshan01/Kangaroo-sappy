import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

# Page config
st.set_page_config(page_title="People Counter", layout="centered")

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
            
            count = 0
            
            # Count person class (class 0)
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    count += 1
            
            count = min(count, 5)
            self.detection_history.append(count)
            
            # Smooth detection with median
            if len(self.detection_history) > 0:
                self.person_count = int(np.median(self.detection_history))
        
        # Return frame as-is (no drawing)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("👥 People Counter")
st.markdown("### Background detection with audio alerts")

# WebRTC streamer (hidden video)
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

st.markdown("---")

# Status display
if ctx.video_processor:
    count_placeholder = st.empty()
    audio_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Update display
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Display count
            count_placeholder.metric(
                label="People Detected", 
                value=current_count,
                delta=None
            )
            
            # Play audio when count changes
            if current_count != ctx.video_processor.last_audio_count and current_count > 0:
                if current_count in AUDIO_FILES:
                    audio_file = AUDIO_FILES[current_count]
                    try:
                        audio_placeholder.audio(audio_file, autoplay=True)
                        ctx.video_processor.last_audio_count = current_count
                        status_placeholder.success(f"🔊 Playing audio for {current_count} {'person' if current_count == 1 else 'people'}")
                    except:
                        status_placeholder.error(f"❌ Audio file {audio_file} not found")
            elif current_count == 0:
                ctx.video_processor.last_audio_count = 0
                audio_placeholder.empty()
                status_placeholder.info("👀 Waiting for people...")
        
        time.sleep(0.5)
else:
    st.info("👆 Click **START** to begin detection")
    st.markdown("""
    **How it works:**
    - Camera runs in background
    - Detects people automatically
    - Plays audio when count changes
    - No video display needed
    """)

st.markdown("---")
st.caption("*Powered by YOLOv8*")
