# video_preview.py
import yt_dlp
import streamlit as st
from .utils import format_duration, format_size, ModelConfig, is_playlist_url, extract_playlist_info, VideoUtility

# video_preview.py
import yt_dlp
import streamlit as st
from .utils import format_duration, format_size, ModelConfig, is_playlist_url, extract_playlist_info, VideoUtility

class VideoPreview:
    def __init__(self):
        self.processing_speed = {model['key']: model['speed_factor'] 
                               for model in ModelConfig.MODELS.values()}

    @st.cache_data
    def get_video_info(_self, url):
        try:
            cleaned_url = VideoUtility.get_clean_video_url(url)
            
            # First attempt - just get basic info without format
            basic_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'format': None,  # No format specification
                'ignoreerrors': True,
                'no_color': True,
                'cookies-from-browser': None
            }
            
            try:
                with yt_dlp.YoutubeDL(basic_opts) as ydl:
                    info = ydl.extract_info(cleaned_url, download=False)
                    if not info:
                        # Second attempt with different options
                        alt_opts = {
                            'quiet': True,
                            'no_warnings': True,
                            'extract_flat': False,
                            'format': None,
                            'ignoreerrors': True,
                            'no_color': True,
                            'youtube_include_dash_manifest': False,
                            'extractor_args': {'youtube': {'skip': ['dash', 'hls']}}
                        }
                        with yt_dlp.YoutubeDL(alt_opts) as ydl2:
                            info = ydl2.extract_info(cleaned_url, download=False)
                    
                    if not info:
                        raise Exception("Could not extract video information")
                    
                    # Process the info we got
                    return {
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration') or 0,
                        'channel': info.get('channel', info.get('uploader', 'Unknown')),
                        'view_count': info.get('view_count', 0),
                        'upload_date': info.get('upload_date', 'Unknown'),
                        'filesize': info.get('filesize_approx', info.get('filesize', 0)),
                        'thumbnail': info.get('thumbnail', info.get('thumbnails', [{'url': None}])[0].get('url')),
                        'is_short': VideoUtility.is_shorts_url(url),
                        'video_id': info.get('id', VideoUtility.get_video_id(url)),
                        'original_url': url
                    }
            
            except Exception as e:
                if "Requested format is not available" in str(e):
                    # Final attempt with absolute minimal options
                    minimal_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': True,
                        'format': None,
                        'skip_download': True,
                        'ignoreerrors': True,
                        'no_color': True
                    }
                    with yt_dlp.YoutubeDL(minimal_opts) as ydl3:
                        info = ydl3.extract_info(cleaned_url, download=False)
                        if info:
                            return {
                                'title': info.get('title', 'Unknown'),
                                'duration': info.get('duration') or 0,
                                'channel': info.get('channel', info.get('uploader', 'Unknown')),
                                'view_count': info.get('view_count', 0),
                                'upload_date': info.get('upload_date', 'Unknown'),
                                'filesize': 0,  # Skip filesize for minimal info
                                'thumbnail': info.get('thumbnail'),
                                'is_short': VideoUtility.is_shorts_url(url),
                                'video_id': info.get('id', VideoUtility.get_video_id(url)),
                                'original_url': url
                            }
                raise  # Re-raise if not a format issue
                
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
            elif "Requested format is not available" in error_msg:
                raise Exception("Could not access this video format. The video might be region-restricted or require special access.")
            else:
                raise Exception(f"Error fetching video info: {error_msg}")
            
            
    def estimate_processing_time(self, duration, model_key):
        base_time = duration * self.processing_speed.get(model_key, 1)
        overhead = 30
        if duration < 60:
            overhead += 15
        return base_time + overhead

    def render_preview(self, url, model_key):
        try:
            if is_playlist_url(url):
                return self.render_playlist_preview(url, model_key)
            else:
                return self.render_video_preview(url, model_key)
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
            return None

    def render_playlist_preview(self, url, model_key):
        try:
            playlist_info = extract_playlist_info(url)
            
            model_display_name = next(
                (name for name, config in ModelConfig.MODELS.items() 
                 if config['key'] == model_key),
                model_key.title()
            )
            
            st.markdown("### 📝 Playlist Information")
            st.markdown(f"**Title:** {playlist_info['title']}")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Videos", playlist_info['video_count'])
            with m2:
                st.metric("Total Duration", format_duration(playlist_info['total_duration']))
            with m3:
                est_time = ModelConfig.estimate_batch_processing_time(
                    playlist_info['total_duration'],
                    model_key
                )
                st.metric("Est. Processing Time", format_duration(est_time))

            st.markdown("### 📋 Videos in Playlist")
            video_data = []
            for idx, video in enumerate(playlist_info['videos'], 1):
                duration = video.get('duration', 0)
                if duration is None:
                    duration = 0
                    
                video_data.append({
                    "№": idx,
                    "Title": video['title'],
                    "Duration": format_duration(duration)
                })
            
            st.table(video_data)

            if playlist_info['total_duration'] > 7200:
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

    def render_video_preview(self, url, model_key):
        try:
            info = self.get_video_info(url)
            
            st.markdown("### 🎥 Video Information")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if info['thumbnail']:
                    st.image(info['thumbnail'])
            
            with col2:
                st.markdown(f"**Title:** {info['title']}")
                st.markdown(f"**Channel:** {info['channel']}")
                
                if info.get('is_short'):
                    st.markdown("**Type:** 📱 YouTube Shorts")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Duration", format_duration(info['duration']))
                with m2:
                    st.metric("File Size", format_size(info['filesize']))
                with m3:
                    st.metric("Views", f"{info['view_count']:,}")

            est_time = self.estimate_processing_time(info['duration'], model_key)
            st.info(f"⏱️ Estimated processing time: {format_duration(est_time)}")

            if info.get('is_short'):
                st.info("ℹ️ YouTube Shorts detected - Processing might take a bit longer")
            elif info['duration'] > 3600:
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