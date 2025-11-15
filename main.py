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
# PAGE CONFIG - Minimal UI
# ----------------------------
st.set_page_config(
    page_title="People Counter",
    page_icon="👥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit branding and menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

# ----------------------------
# AUDIO FILES - Preload as base64
# ----------------------------
@st.cache_data
def load_audio_base64():
    audio_data = {}
    for count, file_path in AUDIO_FILES.items():
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            audio_data[count] = base64.b64encode(audio_bytes).decode()
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return audio_data

AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3", 
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

audio_base64 = load_audio_base64()

# ----------------------------
# AUDIO PLAYER COMPONENT
# ----------------------------
def audio_player(audio_key=None):
    """Creates an audio player that autoplays and auto-cleans up"""
    if audio_key and audio_key in audio_base64:
        audio_html = f"""
        <audio id="peopleCounterAudio" autoplay onended="this.remove()">
            <source src="data:audio/mp3;base64,{audio_base64[audio_key]}" type="audio/mp3">
        </audio>
        <script>
            var allAudio = document.querySelectorAll('audio');
            allAudio.forEach(function(audio) {{
                if (audio.id !== 'peopleCounterAudio') {{
                    audio.pause();
                    audio.currentTime = 0;
                    audio.remove();
                }}
            }});
            setTimeout(function() {{
                var audioElem = document.getElementById('peopleCounterAudio');
                if (audioElem) {{
                    audioElem.remove();
                }}
            }}, 5000);
        </script>
        """
        st.components.v1.html(audio_html, height=0)

# ----------------------------
# YOLO PERSON DETECTOR
# ----------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=5)
        self.last_announced_count = 0
        self.cooldown_frames = 0
        self.cooldown_length = 15
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % 2 == 0:
            results = model(
                img,
                verbose=False,
                imgsz=640,
                conf=0.4,
                device='cpu'
            )
            
            count = 0
            if results[0].boxes is not None:
                count = sum(int(box.cls[0]) == 0 for box in results[0].boxes)
            count = min(count, 5)
            
            self.detection_history.append(count)
            if self.detection_history:
                self.person_count = int(np.median(self.detection_history))
            
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def can_announce(self, current_count):
        if (current_count != self.last_announced_count and 
            current_count > 0 and 
            self.cooldown_frames == 0):
            self.last_announced_count = current_count
            self.cooldown_frames = self.cooldown_length
            return True
        return False

# ----------------------------
# STREAMLIT UI - Material Design Minimal
# ----------------------------

# Custom CSS for Material Design circular buttons
st.markdown("""
<style>
    /* Circular toggle button */
    .stCheckbox > div > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .stCheckbox > label > div:first-child {
        background-color: #1a73e8;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border: none;
    }
    
    .stCheckbox > label > div:first-child:hover {
        background-color: #0d47a1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transform: scale(1.05);
    }
    
    .stCheckbox > label {
        font-size: 0 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 300 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        opacity: 0.8;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Center everything */
    .css-1d391kg, .css-12oz5g7 {
        max-width: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with audio ON by default
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'last_ui_count' not in st.session_state:
    st.session_state.last_ui_count = 0

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Title
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem; font-weight: 300;'>👥 People Counter</h1>", 
                unsafe_allow_html=True)
    
    # People count display
    count_display = st.empty()
    
    # WebRTC streamer
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
    
    # Circular audio toggle button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 10px; font-size: 14px; opacity: 0.8;'>Audio Alerts</p>", 
                unsafe_allow_html=True)
    
    audio_enabled = st.checkbox(
        "🔊",
        value=st.session_state.audio_enabled,
        help="Toggle audio alerts",
        key="audio_toggle"
    )
    
    # Update session state
    st.session_state.audio_enabled = audio_enabled

# Status indicator
status_container = st.empty()

# Main processing loop
if ctx.video_processor:
    audio_placeholder = st.empty()
    
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Update display with Material Design style
            with col2:
                count_display.metric(
                    label="People Detected", 
                    value=current_count
                )
            
            # Check if we should play audio
            should_play = (ctx.video_processor.can_announce(current_count) and 
                          st.session_state.audio_enabled)
            
            if should_play and current_count in audio_base64:
                # Use the audio player component
                with audio_placeholder:
                    audio_player(current_count)
                
                # Show brief status
                with status_container:
                    st.success(f"Announced: {current_count} people")
                    time.sleep(1)
                    status_container.empty()
                
                st.session_state.last_ui_count = current_count
            elif current_count == 0 and st.session_state.last_ui_count != 0:
                st.session_state.last_ui_count = 0
            elif current_count == st.session_state.last_ui_count and current_count > 0:
                # Quiet tracking, no status update to keep UI clean
                pass
        
        time.sleep(0.3)
