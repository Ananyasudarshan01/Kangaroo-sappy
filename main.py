import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import time
import base64

# ----------------------------
# 🔧 PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="People Counter", layout="centered")

# ----------------------------
# 🔍 LOAD YOLO MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

# ----------------------------
# 🔊 AUDIO FILES
# ----------------------------
AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3",
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

# ----------------------------
# 🎵 PLAY AUDIO (SAFE HTML VERSION)
# ----------------------------
def play_audio(file_path):
    """Play audio using inline HTML to avoid Streamlit caching or UI issues."""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        unique = str(time.time()).replace(".", "")
        audio_html = f"""
        <audio id="player_{unique}" autoplay>
            <source src="data:audio/mp3;base64,{b64}?v={unique}" type="audio/mp3">
        </audio>
        """
        # Inject small HTML audio player invisibly under Streamlit widgets
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error playing {file_path}: {e}")

# ----------------------------
# 🧠 VIDEO PROCESSOR
# ----------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=3)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Run detection every 2nd frame for performance
        if self.frame_count % 2 == 0:
            results = model(
                img,
                verbose=False,
                imgsz=640,
                conf=0.4,
                device='cpu'
            )
            
            # Count 'person' class (class 0)
            count = sum(int(box.cls[0]) == 0 for box in results[0].boxes)
            count = min(count, 5)
            
            self.detection_history.append(count)
            if self.detection_history:
                self.person_count = int(np.median(self.detection_history))
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# 🖥️ STREAMLIT UI
# ----------------------------
st.title("👥 People Counter")
st.markdown("### Detects people in camera feed and plays sound alerts")

# WebRTC video (hidden feed)
ctx = webrtc_streamer(
    key="people-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PersonDetector,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
)

st.markdown("---")

# ----------------------------
# ⚙️ MAIN LOOP
# ----------------------------
if ctx.video_processor:
    count_placeholder = st.empty()
    status_placeholder = st.empty()
    last_played_count = 0
    
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Show metric
            count_placeholder.metric("People Detected", current_count)
            
            # Play new audio when count changes
            if current_count != last_played_count and current_count > 0:
                if current_count in AUDIO_FILES:
                    play_audio(AUDIO_FILES[current_count])
                    status_placeholder.success(
                        f"🔊 Playing audio for {current_count} "
                        f"{'person' if current_count == 1 else 'people'}"
                    )
                last_played_count = current_count
            elif current_count == 0 and last_played_count != 0:
                last_played_count = 0
                status_placeholder.info("👀 Waiting for people...")
        
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to begin detection")
    st.markdown("""
    **How it works:**
    - Uses your webcam (not displayed)
    - Detects how many people appear
    - Plays audio when count changes
    """)

st.markdown("---")
st.caption("*Powered by YOLOv8 and Streamlit WebRTC*")
