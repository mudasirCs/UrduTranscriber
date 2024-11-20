import yt_dlp
import streamlit as st
from .utils import format_duration, format_size, ModelConfig


class VideoPreview:


    def __init__(self):
        self.processing_speed = {model['key']: model['speed_factor'] 
                               for model in ModelConfig.MODELS.values()}

    @st.cache_data  # Cache video info
    def get_video_info(_self, url):
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'channel': info.get('channel', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'filesize': info.get('filesize_approx', 0),
                    'thumbnail': info.get('thumbnail', None)
                }
        except Exception as e:
            raise Exception(f"Error fetching video info: {str(e)}")

    def estimate_processing_time(self, duration, model_name):
        """Estimate processing time in seconds"""
        base_time = duration * self.processing_speed.get(model_name, 1)
        overhead = 30  # Additional time for download and document generation
        return base_time + overhead

    def render_preview(self, url, selected_model):
        try:
            info = self.get_video_info(url)
            with st.expander("Video Preview", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if info['thumbnail']:
                        st.image(info['thumbnail'])
                
                with col2:
                    st.markdown(f"### {info['title']}")
                    st.markdown(f"**Channel:** {info['channel']}")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Duration", format_duration(info['duration']))
                    with m2:
                        st.metric("File Size", format_size(info['filesize']))
                    with m3:
                        st.metric("Views", f"{info['view_count']:,}")

                est_time = self.estimate_processing_time(info['duration'], selected_model)
                st.info(f"Estimated processing time: {format_duration(est_time)}")

                if info['duration'] > 3600:  # Warning for videos longer than 1 hour
                    st.warning("""
                        ⚠️ Long video detected! Consider:
                        - Using a faster model
                        - Making sure your computer won't sleep
                        - Being patient 😊
                    """)
                
                return info
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
            return None