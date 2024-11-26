# video_preview.py
import yt_dlp
import streamlit as st
from .utils import format_duration, format_size, ModelConfig, is_playlist_url, extract_playlist_info, VideoUtility

class VideoPreview:
    def __init__(self):
        self.processing_speed = {model['key']: model['speed_factor'] 
                               for model in ModelConfig.MODELS.values()}

    @st.cache_data  # Cache video info
    def get_video_info(_self, url):
        try:
            # Clean URL to get just the video URL without playlist parameters
            cleaned_url = VideoUtility.get_clean_video_url(url)
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,  # Get full info first
                'format': 'best',  # Request best format for proper duration
                'ignoreerrors': True,
                'no_color': True,
                'cookies-from-browser': None,  # Don't try to load cookies
                'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}  # Skip these formats for faster info extraction
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # First attempt with full info extraction
                    info = ydl.extract_info(cleaned_url, download=False)
                except Exception:
                    # If first attempt fails, try with simpler options
                    ydl_opts['extract_flat'] = True
                    ydl_opts['format'] = None
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        info = ydl2.extract_info(cleaned_url, download=False)
                
                if not info:
                    raise Exception("Could not extract video information")
                
                # Extract information with fallbacks
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration') or 0,  # Handle None case
                    'channel': info.get('channel', info.get('uploader', 'Unknown')),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'filesize': info.get('filesize_approx', info.get('filesize', 0)),
                    'thumbnail': info.get('thumbnail', info.get('thumbnails', [{'url': None}])[0].get('url')),
                    'is_short': VideoUtility.is_shorts_url(url),
                    'video_id': info.get('id', VideoUtility.get_video_id(url)),
                    'original_url': url  # Keep original URL for reference
                }
                
        except Exception as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                raise Exception("This video is unavailable or private")
            elif "Sign in" in error_msg:
                raise Exception("This video requires age verification or sign-in")
            elif "Private video" in error_msg:
                raise Exception("This is a private video")
            elif "This live event has ended" in error_msg:
                raise Exception("This live stream has ended")
            else:
                raise Exception(f"Error fetching video info: {error_msg}")

    def estimate_processing_time(self, duration, model_name):
        """Estimate processing time in seconds"""
        base_time = duration * self.processing_speed.get(model_name, 1)
        overhead = 30  # Additional time for download and document generation
        
        # Add extra overhead for shorts (they often need additional processing)
        if duration < 60:  # Typical shorts duration
            overhead += 15
            
        return base_time + overhead

    def render_preview(self, url, selected_model):
        try:
            if is_playlist_url(url):
                return self.render_playlist_preview(url, selected_model)
            else:
                return self.render_video_preview(url, selected_model)
                
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
            return None

    def render_playlist_preview(self, url, selected_model):
        """Render playlist preview"""
        try:
            playlist_info = extract_playlist_info(url)
            
            # Main container for playlist info
            st.markdown("### 📝 Playlist Information")
            st.markdown(f"**Title:** {playlist_info['title']}")
            
            # Display metrics in columns
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Videos", playlist_info['video_count'])
            with m2:
                st.metric("Total Duration", format_duration(playlist_info['total_duration']))
            with m3:
                est_time = ModelConfig.estimate_batch_processing_time(
                    playlist_info['total_duration'],
                    selected_model
                )
                st.metric("Est. Processing Time", format_duration(est_time))

            # Show video list in a clean table format
            st.markdown("### 📋 Videos in Playlist")
            
            # Create a DataFrame for better display
            video_data = []
            for idx, video in enumerate(playlist_info['videos'], 1):
                # Handle potential missing durations
                duration = video.get('duration', 0)
                if duration is None:
                    duration = 0
                    
                video_data.append({
                    "№": idx,
                    "Title": video['title'],
                    "Duration": format_duration(duration)
                })
            
            # Display as a styled table
            st.table(video_data)

            if playlist_info['total_duration'] > 7200:  # 2 hours
                st.warning("""
                    ⚠️ Long playlist detected! Consider:
                    - Using a faster model
                    - Making sure your computer won't sleep
                    - Being patient 😊
                """)
            
            return playlist_info
            
        except Exception as e:
            st.error(f"Error loading playlist: {str(e)}")
            return None

    def render_video_preview(self, url, selected_model):
        """Render single video preview"""
        try:
            info = self.get_video_info(url)
            
            # Main container for video info
            st.markdown("### 🎥 Video Information")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if info['thumbnail']:
                    st.image(info['thumbnail'])
            
            with col2:
                st.markdown(f"**Title:** {info['title']}")
                st.markdown(f"**Channel:** {info['channel']}")
                
                # Add shorts indicator if it's a shorts video
                if info.get('is_short'):
                    st.markdown("**Type:** 📱 YouTube Shorts")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Duration", format_duration(info['duration']))
                with m2:
                    st.metric("File Size", format_size(info['filesize']))
                with m3:
                    st.metric("Views", f"{info['view_count']:,}")

            est_time = self.estimate_processing_time(info['duration'], selected_model)
            st.info(f"⏱️ Estimated processing time: {format_duration(est_time)}")

            # Warnings based on video type and duration
            if info.get('is_short'):
                st.info("""
                    ℹ️ YouTube Shorts detected:
                    - Processing might take a bit longer
                    - Quality may vary due to video format
                """)
            elif info['duration'] > 3600:  # Warning for videos longer than 1 hour
                st.warning("""
                    ⚠️ Long video detected! Consider:
                    - Using a faster model
                    - Making sure your computer won't sleep
                    - Being patient 😊
                """)
            
            return info
            
        except Exception as e:
            st.error(f"Error loading video preview: {str(e)}")
            return None