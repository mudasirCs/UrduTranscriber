import streamlit as st
from src.video_preview import VideoPreview
from src.transcriber import TranscriptionManager
from src.styles import CustomCSS
from src.utils import ModelConfig
import os

# Must be first Streamlit command
st.set_page_config(
    page_title="YouTube Transcriber",
    page_icon="📝",
    layout="wide"
)

# Apply custom CSS
st.markdown(CustomCSS.STYLES, unsafe_allow_html=True)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = TranscriptionManager()
    st.session_state.preview = VideoPreview()

def main():
    st.markdown('<h1 class="main-title">YouTube Urdu Video Transcriber', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Model Settings")
        model_option = st.selectbox(
            "Select Model",
            options=list(ModelConfig.MODELS.keys()),
            index=1
        )
        model_info = ModelConfig.MODELS[model_option]
        
        st.markdown(f"""
        **Model Details:**
        - {model_info['description']}
        - Size: {model_info['size']}
        - Download: {model_info['download_size']}
        """)

    # Main content
    url = st.text_input("Enter YouTube URL")

    if url:
        try:
            # Show video preview
            video_info = st.session_state.preview.render_preview(
                url, 
                model_info['key']
            )
            
            if video_info:
                if st.button("Start Transcription"):
                    with st.spinner("Processing..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            doc_path = st.session_state.manager.process_video(
                                url,
                                model_info['key'],lambda p, s: (progress_bar.progress(p), status_text.text(s))
                            )
                            
                            if doc_path:
                                st.success("✅ Transcription completed!")
                                with open(doc_path, 'rb') as f:
                                    st.download_button(
                                        "📄 Download Transcript",
                                        f,
                                        file_name=os.path.basename(doc_path),
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    )
                        except Exception as e:
                            st.error(f"Transcription failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

    # Recent transcriptions
    st.markdown("---")
    st.header("Recent Transcriptions")
    
    recent_files = st.session_state.manager.list_recent_transcripts()
    if recent_files:
        for file_path in recent_files:
            with open(file_path, 'rb') as f:
                st.download_button(
                    f"📄 {file_path.name}",
                    f,
                    file_name=file_path.name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=str(file_path)
                )
    else:
        st.info("No recent transcriptions found")

if __name__ == "__main__":
    main()