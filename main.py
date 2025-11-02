import base64
import streamlit.components.v1 as components

def play_audio(file_path):
    """Embed base64 MP3 audio that auto-plays reliably across re-renders."""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        unique_id = str(time.time()).replace('.', '')
        
        # HTML + JS for guaranteed playback
        html_code = f"""
        <script>
        var oldAudio = document.getElementById("audio_{unique_id}");
        if (oldAudio) {{
            oldAudio.pause();
            oldAudio.remove();
        }}
        var audio = document.createElement("audio");
        audio.id = "audio_{unique_id}";
        audio.src = "data:audio/mp3;base64,{b64}";
        audio.autoplay = true;
        audio.volume = 1.0;
        document.body.appendChild(audio);
        audio.play().catch(e => console.log("Autoplay blocked:", e));
        </script>
        """
        components.html(html_code, height=0, width=0)
    except Exception as e:
        st.error(f"❌ Error playing {file_path}: {e}")
