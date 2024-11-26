import concurrent.futures
import threading
import time
import pandas as pd
import streamlit as st
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import yt_dlp
from .utils import extract_playlist_info, format_duration, get_timestamp
import re
import torch
import shutil
from urllib.error import HTTPError

@dataclass
class VideoTask:
    idx: int
    url: str
    title: str
    duration: int
    status: str = 'pending'  # 'pending', 'downloading', 'transcribing', 'completed', 'failed'
    error: Optional[str] = None
    transcript_path: Optional[str] = None
    audio_path: Optional[str] = None
    last_attempt: Optional[str] = None
    attempt_count: int = 0
    download_size: int = 0
    downloaded_size: int = 0
    download_speed: float = 0
    processing_progress: float = 0
    stop_requested: bool = False
    channel: str = 'Unknown'  # Added default value
    video_id: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: int = 0
    upload_date: str = 'Unknown'

    
class PlaylistProcessor:
    def __init__(self, transcription_manager=None):
        """Initialize PlaylistProcessor with optional transcription manager"""
        self.manager = transcription_manager
        self.tasks: Dict[str, VideoTask] = {}
        self.lock = threading.Lock()
        self.current_playlist_dir = None
        self.stop_event = threading.Event()

    def setup_playlist_directory(self, playlist_info):
        """Create and setup playlist directory structure"""
        timestamp = get_timestamp()
        safe_title = self.sanitize_text(playlist_info['title']).replace(" ", "_")[:50]
        base_dir = self.manager.dirs['output'] / f"playlist_{timestamp}_{safe_title}"
        
        # Create directory structure
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "audio").mkdir(exist_ok=True)
        (base_dir / "transcripts").mkdir(exist_ok=True)
        (base_dir / "failed").mkdir(exist_ok=True)
        
        self.current_playlist_dir = base_dir
        return base_dir

    def get_audio_path(self, video_id):
        """Get path for audio file"""
        return self.current_playlist_dir / "audio" / f"{video_id}.mp3"

    def get_transcript_path(self, video_id, title):
        """Get path for transcript file"""
        safe_title = self.sanitize_text(title).replace(" ", "_")[:50]
        return self.current_playlist_dir / "transcripts" / f"{video_id}_{safe_title}.docx"

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text to be XML compatible and safe for filenames"""
        if not text:
            return ""
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if char in '\n\t' or (ord(char) >= 32 and ord(char) != 127))
        # Replace any remaining invalid characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Replace filesystem unsafe characters
        text = re.sub(r'[<>:"/\\|?*]', '_', text)
        # Remove leading/trailing spaces and dots
        text = text.strip('. ')
        return text

    def process_playlist(self, playlist_url: str, model_name: str = "base", retry_videos: List[str] = None, retry_only: bool = False):
        """Process playlist processing"""
        try:
            if not self.manager:
                raise Exception("Transcription manager not set")

            # Get playlist info if not retrying specific videos
            if not retry_videos and not retry_only:
                playlist_info = extract_playlist_info(playlist_url)
                total_videos = len(playlist_info['videos'])
                
                if total_videos == 0:
                    raise Exception("No valid videos found in playlist")
                
                # Create tasks for each video
                self.tasks.clear()
                for idx, video in enumerate(playlist_info['videos'], 1):
                    video_id = video['url'].split('v=')[-1].split('&')[0]

                    # Create task with all available information
                    self.tasks[video_id] = VideoTask(
                        idx=idx,
                        url=video['url'],
                        title=video.get('title', 'Unknown'),
                        duration=video.get('duration', 0),
                        channel=video.get('channel', 'Unknown'),  # Set channel explicitly
                        video_id=video_id,
                        thumbnail=video.get('thumbnail'),
                        view_count=video.get('view_count', 0),
                        upload_date=video.get('upload_date', 'Unknown')
                    )
            else:
                # Handle retry logic
                if retry_only:
                    # Get list of failed video IDs
                    failed_videos = [vid_id for vid_id, task in self.tasks.items() 
                                if task.status == 'failed']
                    retry_videos = failed_videos
                
                if not retry_videos:
                    st.warning("No failed videos to retry")
                    return None
                    
                playlist_info = {'title': 'Retry Processing'}
                total_videos = len(retry_videos)
                
                # Reset status for retry videos
                for video_id in retry_videos:
                    if video_id in self.tasks:
                        task = self.tasks[video_id]
                        task.status = 'pending'
                        task.error = None
                        task.last_attempt = None
                        task.attempt_count = 0
                        task.download_progress = 0
                        task.processing_progress = 0

            # Ensure we have videos to process
            if not self.tasks:
                st.warning("No videos to process")
                return None

            # Get the list of videos to process
            videos_to_process = retry_videos if retry_videos else list(self.tasks.keys())
            total_videos = len(videos_to_process)

            if total_videos == 0:
                st.warning("No videos to process")
                return None

            # Create status displays
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_table = st.empty()
            
            # Process videos
            completed = 0
            failed = 0
            
            for idx, video_id in enumerate(videos_to_process, 1):
                if self.stop_event.is_set():
                    st.warning("Processing stopped by user")
                    break

                task = self.tasks.get(video_id)
                if not task:
                    continue
                
                try:
                    # Update status display
                    self.display_status_table(status_table)
                    
                    # Process video
                    st.info(f"Processing video {idx}/{total_videos}: {task.title}")
                    result = self.process_single_video(video_id, task, model_name)
                    
                    if result and result.get('success'):
                        completed += 1
                        st.success(f"Successfully processed: {task.title}")
                    else:
                        failed += 1
                        st.error(f"Failed to process: {task.title}")
                    
                except Exception as e:
                    failed += 1
                    st.error(f"Error processing video {task.title}: {str(e)}")
                
                # Update progress
                progress = float(idx) / float(total_videos)  # Explicit float conversion
                progress_bar.progress(progress)
                status_text.text(f"Processed: {idx}/{total_videos} videos")
                self.display_status_table(status_table)
            
            # Create summary if any videos were processed
            if completed > 0 or failed > 0:
                # Only create new directory for full playlist processing or if it doesn't exist
                if not retry_only or not hasattr(self, 'current_playlist_dir'):
                    timestamp = get_timestamp()
                    safe_title = self.sanitize_text(playlist_info['title']).replace(" ", "_")[:50]
                    output_dir = self.manager.dirs['output'] / f"playlist_{timestamp}_{safe_title}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.current_playlist_dir = output_dir
                else:
                    output_dir = self.current_playlist_dir
                
                # Move successful transcripts to playlist directory
                successful_tasks = [task for task in self.tasks.values() 
                                if task.status == 'completed' and task.transcript_path]
                
                for task in successful_tasks:
                    if task.transcript_path:
                        source = Path(task.transcript_path)
                        if source.exists():
                            dest = output_dir / source.name
                            shutil.move(str(source), str(dest))
                            task.transcript_path = str(dest)
                
                # Create and save summary
                summary_path = self.create_summary(playlist_info, output_dir)
                
                return {
                    'summary': summary_path,
                    'completed': completed,
                    'failed': failed,
                    'output_dir': output_dir
                }
            else:
                st.warning("No videos were processed successfully")
                return None
                
        except Exception as e:
            st.error(f"Error processing playlist: {str(e)}")
            raise

    def process_single_video(self, video_id: str, task: VideoTask, model_name: str):
        """Process a single video with proper metadata handling"""
        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                self.update_task_status(video_id, 'processing')
                task.last_attempt = datetime.now().isoformat()
                
                # Download and process
                audio_info = None
                try:
                    # Get detailed video info before downloading
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,  # Get full info
                        'format': 'best',
                        'ignoreerrors': True,
                        'no_color': True
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        video_info = ydl.extract_info(task.url, download=False)
                        
                        # Update task with accurate information
                        if video_info:
                            task.title = video_info.get('title', task.title)
                            task.duration = video_info.get('duration', task.duration)
                            task.channel = video_info.get('channel', video_info.get('uploader', 'Unknown'))
                            task.view_count = video_info.get('view_count', task.view_count)
                            task.upload_date = video_info.get('upload_date', task.upload_date)
                            task.thumbnail = video_info.get('thumbnail', task.thumbnail)

                    
                    # Download audio
                    st.info(f"Downloading: {task.title} (Attempt {attempt + 1}/{max_retries})")
                    audio_info = self.manager.download_audio(
                        task.url,
                        max_retries=3,
                        retry_delay=5
                    )
                    
                    if not audio_info:
                        raise Exception("Failed to download audio")
                    
                    # Process video with more GPU memory management
                    st.info(f"Transcribing: {task.title}")
                    
                    # Clear GPU memory before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Create a progress callback for this video
                    def progress_callback(progress: float, status: str):
                        if isinstance(progress, (int, float)):
                            task.processing_progress = float(progress)
                        st.text(status)
                    
                    # Pass full video information to process_single_video
                    doc_path = self.manager.process_single_video(
                        audio_info['path'],
                        task.url,
                        task.title,
                        progress_callback,
                        metadata={
                            'duration': task.duration,
                            'channel': task.channel,
                            'title': task.title,
                            'view_count': task.view_count,
                            'upload_date': task.upload_date
                        }
                    )
                    
                    if doc_path:
                        self.update_task_status(video_id, 'completed', transcript_path=doc_path)
                        st.success(f"Successfully processed: {task.title}")
                        return {'success': True, 'path': doc_path}
                    else:
                        raise Exception("Failed to generate transcript")
                    
                except Exception as e:
                    error_msg = str(e)
                    if "cannot reshape tensor" in error_msg:
                        st.warning("GPU memory issue detected. Cleaning up and retrying...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(retry_delay * 2)
                    
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed for {task.title}: {error_msg}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.update_task_status(video_id, 'failed', error_msg)
                        return {'success': False, 'error': error_msg}
                    
                finally:
                    # Cleanup audio file
                    if audio_info and 'path' in audio_info:
                        try:
                            Path(audio_info['path']).unlink(missing_ok=True)
                        except Exception:
                            pass
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                else:
                    self.update_task_status(video_id, 'failed', str(e))
                    st.error(f"Failed to process {task.title}: {str(e)}")
                    return {'success': False, 'error': str(e)}

        st.error(f"Failed to process {task.title}: Max retries exceeded")
        return {'success': False, 'error': 'Max retries exceeded'}


    def display_status_table(self, container):
        """Display status table with action buttons"""
        with container:
            st.markdown("### 📊 Processing Status")
            
            # Create header for the table
            header_cols = st.columns([0.5, 2, 1, 1, 1, 1.5, 1])
            with header_cols[0]:
                st.markdown("**#**")
            with header_cols[1]:
                st.markdown("**Title**")
            with header_cols[2]:
                st.markdown("**Duration**")
            with header_cols[3]:
                st.markdown("**Status**")
            with header_cols[4]:
                st.markdown("**Progress**")
            with header_cols[5]:
                st.markdown("**Last Update**")
            with header_cols[6]:
                st.markdown("**Actions**")
            
            st.markdown("---")
            
            timestamp = datetime.now().strftime("%H%M%S")
            failed_tasks = []
            
            # Display each video's status
            for video_id, task in sorted(self.tasks.items(), key=lambda x: x[1].idx):
                cols = st.columns([0.5, 2, 1, 1, 1, 1.5, 1])
                
                # Index
                with cols[0]:
                    st.text(f"{task.idx}")
                
                # Title with truncation
                with cols[1]:
                    full_title = task.title
                    display_title = full_title[:40] + "..." if len(full_title) > 40 else full_title
                    st.markdown(f"{display_title}")
                
                # Duration
                with cols[2]:
                    st.text(format_duration(task.duration))
                
                # Status with color coding
                with cols[3]:
                    status_colors = {
                        'pending': ('🔄', 'gray'),
                        'downloading': ('⬇️', 'blue'),
                        'transcribing': ('🔄', 'blue'),
                        'completed': ('✅', 'green'),
                        'failed': ('❌', 'red')
                    }
                    emoji, color = status_colors.get(task.status, ('❓', 'gray'))
                    st.markdown(f"<span style='color: {color}'>{emoji} {task.status}</span>", 
                            unsafe_allow_html=True)
                
                # Progress
                with cols[4]:
                    if task.status != 'pending':
                        st.progress(task.processing_progress)
                    else:
                        st.text("Waiting...")
                
                # Last update time
                with cols[5]:
                    if task.last_attempt:
                        try:
                            last_time = datetime.fromisoformat(task.last_attempt)
                            time_str = last_time.strftime("%H:%M:%S")
                            st.text(time_str)
                        except:
                            st.text("--:--:--")
                    else:
                        st.text("--:--:--")
                
                # Action buttons
                with cols[6]:
                    if task.status == 'completed' and task.transcript_path:
                        try:
                            with open(task.transcript_path, 'rb') as f:
                                st.download_button(
                                    "📄",
                                    f,
                                    file_name=Path(task.transcript_path).name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"download_{video_id}_{timestamp}"
                                )
                        except Exception as e:
                            st.error("⚠️ Error loading file")
                    
                    elif task.status == 'failed':
                        failed_tasks.append(video_id)
                        if st.button("🔄",
                                key=f"retry_{video_id}_{timestamp}"):
                            st.info(f"Retrying: {task.title}")
                            self.process_playlist(None, "base", [video_id])
            
            # Summary statistics at the bottom
            st.markdown("---")
            stats = self.get_task_statistics()
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Completed", f"{stats['completed']}/{stats['total']}")
            with cols[1]:
                st.metric("Failed", str(stats['failed']))
            with cols[2]:
                st.metric("Processing", str(stats['processing']))
            with cols[3]:
                st.metric("Pending", str(stats['pending']))
            
            # Controls for failed tasks
            if failed_tasks:
                st.markdown("---")
                control_cols = st.columns([2, 1])
                with control_cols[0]:
                    st.error(f"🚨 {len(failed_tasks)} videos failed to process")
                with control_cols[1]:
                    if st.button("🔄 Retry All Failed",
                            key=f"retry_all_failed_{timestamp}"):
                        st.info("Retrying all failed videos...")
                        self.process_playlist(None, "base", failed_tasks)

    def move_failed_audio(self, video_id, audio_path):
        """Move failed audio file to failed directory"""
        if audio_path and Path(audio_path).exists():
            try:
                failed_dir = self.current_playlist_dir / "failed"
                failed_path = failed_dir / f"{video_id}.mp3"
                shutil.move(audio_path, str(failed_path))
                return str(failed_path)
            except Exception as e:
                st.warning(f"Failed to move failed audio file: {str(e)}")
        return None

    def update_task_status(self, video_id: str, status: str, error: Optional[str] = None, 
                          transcript_path: Optional[str] = None, audio_path: Optional[str] = None):
        """Update task status with thread safety"""
        with self.lock:
            if video_id in self.tasks:
                task = self.tasks[video_id]
                task.status = status
                if error:
                    task.error = error
                if transcript_path:
                    task.transcript_path = transcript_path
                if audio_path:
                    task.audio_path = audio_path
                task.last_attempt = datetime.now().isoformat()


    def create_summary(self, playlist_info, output_dir: Path):
        """Create comprehensive summary document"""
        doc = Document()
        
        # Title
        title = doc.add_heading(f"Playlist: {playlist_info['title']}", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Summary info
        doc.add_heading('Playlist Information', level=1)
        summary_table = doc.add_table(rows=6, cols=2)
        summary_table.style = 'Table Grid'
        
        # Get statistics
        stats = self.get_task_statistics()
        error_summary = self.get_error_summary()
        
        # Fill summary table
        rows = summary_table.rows
        rows[0].cells[0].text = 'Total Videos'
        rows[0].cells[1].text = str(stats['total'])
        rows[1].cells[0].text = 'Successfully Processed'
        rows[1].cells[1].text = str(stats['completed'])
        rows[2].cells[0].text = 'Failed Videos'
        rows[2].cells[1].text = str(stats['failed'])
        rows[3].cells[0].text = 'Total Duration'
        rows[3].cells[1].text = format_duration(stats['total_duration'])
        rows[4].cells[0].text = 'Processed Duration'
        rows[4].cells[1].text = format_duration(stats['processed_duration'])
        rows[5].cells[0].text = 'Processing Date'
        rows[5].cells[1].text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Error Summary
        if error_summary:
            doc.add_heading('Error Summary', level=1)
            error_table = doc.add_table(rows=len(error_summary), cols=2)
            error_table.style = 'Table Grid'
            
            for idx, (error_type, count) in enumerate(error_summary.items()):
                row = error_table.rows[idx]
                row.cells[0].text = error_type
                row.cells[1].text = str(count)
        
        # Successful videos
        completed_tasks = [t for t in self.tasks.values() if t.status == 'completed']
        if completed_tasks:
            doc.add_heading('Successfully Processed Videos', level=1)
            for task in completed_tasks:
                p = doc.add_paragraph()
                p.add_run('✓ ').bold = True
                p.add_run(task.title)
                if task.transcript_path:
                    p.add_run(f"\nTranscript: {Path(task.transcript_path).name}")
                if task.audio_path:
                    p.add_run(f"\nAudio: {Path(task.audio_path).name}")
                p.add_run(f"\nDuration: {format_duration(task.duration)}")
        
        # Failed videos
        failed_tasks = [t for t in self.tasks.values() if t.status == 'failed']
        if failed_tasks:
            doc.add_heading('Failed Videos', level=1)
            for task in failed_tasks:
                p = doc.add_paragraph()
                p.add_run('✗ ').bold = True
                p.add_run(task.title)
                p.add_run(f"\nDuration: {format_duration(task.duration)}")
                if task.error:
                    p.add_run(f"\nError: {task.error}")
                if task.attempt_count:
                    p.add_run(f"\nAttempts: {task.attempt_count}")
        
        # Save summary
        summary_path = output_dir / "00_playlist_summary.docx"
        doc.save(str(summary_path))
        return summary_path

    def get_task_statistics(self):
        """Get comprehensive processing statistics"""
        with self.lock:
            stats = {
                'total': len(self.tasks),
                'completed': len([t for t in self.tasks.values() if t.status == 'completed']),
                'failed': len([t for t in self.tasks.values() if t.status == 'failed']),
                'processing': len([t for t in self.tasks.values() 
                                 if t.status in ['processing', 'downloading', 'transcribing']]),
                'pending': len([t for t in self.tasks.values() if t.status == 'pending']),
                'total_duration': sum(task.duration for task in self.tasks.values()),
                'processed_duration': sum(task.duration for task in self.tasks.values() 
                                       if task.status == 'completed')
            }
            return stats

    def get_error_summary(self):
        """Get comprehensive error summary"""
        error_counts = {}
        for task in self.tasks.values():
            if task.error:
                error_type = task.error.split(':')[0].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))