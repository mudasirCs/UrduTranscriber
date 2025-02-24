from datetime import datetime
import os
import re
from pathlib import Path
import ffmpeg
import yt_dlp
from typing import Dict, List, Any, Optional
import streamlit as st
import torch
import shutil
from urllib.parse import urlparse, parse_qs


def format_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration"""
    if not seconds:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size"""
    if not size_bytes:
        return "0B"
        
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def sanitize_filename(title: str) -> str:
    """Create safe filename from title"""
    # Remove or replace unsafe characters
    title = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Remove leading/trailing spaces and dots
    title = title.strip('. ')
    # Replace multiple spaces with single underscore
    title = re.sub(r'\s+', '_', title)
    # Remove any non-ASCII characters
    title = ''.join(char for char in title if ord(char) < 128)
    # Limit length
    return title[:50]

def get_timestamp() -> str:
    """Get current timestamp for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def is_playlist_url(url: str) -> bool:
    """Check if URL is a playlist"""
    playlist_indicators = [
        'playlist?list=',
        '/playlist/'
    ]
    # Only consider it a playlist if it's a pure playlist URL
    return any(indicator in url for indicator in playlist_indicators)

def extract_playlist_info(url: str) -> Dict[str, Any]:
    """Extract playlist information using yt-dlp"""
    if not is_playlist_url(url):
        raise ValueError("Not a valid playlist URL")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'ignoreerrors': True,
        'nocheckcertificate': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError("Could not extract playlist information")
            
            if 'entries' not in info:
                raise ValueError("No playlist entries found")
                
            videos = []
            total_duration = 0
            
            entries = info.get('entries', [])
            if not entries:
                raise ValueError("No videos found in playlist")

            for entry in entries:
                if entry and isinstance(entry, dict):  # Verify entry is valid
                    try:
                        video_info = {
                            'url': f"https://youtube.com/watch?v={entry['id']}",
                            'title': entry.get('title', 'Unknown'),
                            'duration': entry.get('duration', 0),
                            'channel': entry.get('channel', entry.get('uploader', 'Unknown')),
                            'thumbnail': entry.get('thumbnail'),
                            'view_count': entry.get('view_count', 0),
                            'upload_date': entry.get('upload_date', 'Unknown')
                        }
                        videos.append(video_info)
                        total_duration += video_info['duration'] or 0
                    except Exception:
                        continue  # Skip invalid entries
            
            if not videos:
                raise ValueError("No valid videos found in playlist")
            
            return {
                'title': info.get('title', 'Unknown Playlist'),
                'channel': info.get('channel', info.get('uploader', 'Unknown Channel')),
                'videos': videos,
                'total_duration': total_duration,
                'video_count': len(videos),
                'playlist_id': info.get('id')
            }
            
    except Exception as e:
        raise Exception(f"Error extracting playlist info: {str(e)}")

class ModelConfig:
    """Configuration for different Whisper models"""
    MODELS = {
        "Large": {
            "key": "large",
            "size": "1550M parameters",
            "description": "Best accuracy, slowest processing",
            "download_size": "~3 GB",
            "speed_factor": 4.0,
            "memory_requirement": "6GB VRAM"
        },
        "Medium": {
            "key": "medium",
            "size": "769M parameters",
            "description": "High accuracy, slower processing",
            "download_size": "~1.5 GB",
            "speed_factor": 3.0,
            "memory_requirement": "4GB VRAM"
        },
        "Small": {
            "key": "small",
            "size": "244M parameters",
            "description": "Better accuracy, still reasonable speed",
            "download_size": "~466 MB",
            "speed_factor": 2.0,
            "memory_requirement": "3GB VRAM"
        },
        "Base": {
            "key": "base",
            "size": "74M parameters",
            "description": "Good balance of speed and accuracy",
            "download_size": "~142 MB",
            "speed_factor": 1.0,
            "memory_requirement": "2GB VRAM"
        },
        "Tiny": {
            "key": "tiny",
            "size": "39M parameters",
            "description": "Fastest but least accurate",
            "download_size": "~75 MB",
            "speed_factor": 0.5,
            "memory_requirement": "1GB VRAM"
        }
    }

    @staticmethod
    def estimate_batch_processing_time(total_duration: int, model_name: str) -> int:
        """Estimate total processing time for multiple videos"""
        base_factor = {
            'tiny': 0.5,
            'base': 1.0,
            'small': 2.0,
            'medium': 3.0,
            'large': 4.0
        }.get(model_name, 1.0)
        
        processing_time = total_duration * base_factor
        overhead_per_minute = 2
        total_overhead = (total_duration / 60) * overhead_per_minute
        
        return int(processing_time + total_overhead)

    @staticmethod
    def get_model_requirements(model_name: str) -> Dict[str, Any]:
        """Get hardware requirements for a model"""
        requirements = {
            'tiny': {'vram': 1, 'ram': 4},
            'base': {'vram': 2, 'ram': 8},
            'small': {'vram': 3, 'ram': 12},
            'medium': {'vram': 4, 'ram': 16},
            'large': {'vram': 6, 'ram': 24}
        }
        return requirements.get(model_name, {'vram': 2, 'ram': 8})

class VideoUtility:
    """Utility class for video-related functions"""
    
    @staticmethod
    def get_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        if 'watch?v=' in url and '&list=' in url:
            vid_part = url.split('watch?v=')[1]
            return vid_part.split('&')[0]
            
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11})',  # Regular video
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Shortened
            r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embedded
            r'(?:shorts\/)([0-9A-Za-z_-]{11})'  # Shorts
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def get_clean_video_url(url: str) -> str:
        """Convert any YouTube URL to a simple video URL"""
        video_id = VideoUtility.get_video_id(url)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return url
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate YouTube URL"""
        patterns = [
            r'^https?:\/\/(www\.)?youtube\.com\/(watch\?v=|playlist\?list=|embed\/|shorts\/)',
            r'^https?:\/\/youtu\.be\/'
        ]
        return any(re.match(pattern, url) for pattern in patterns)
    
    @staticmethod
    def is_shorts_url(url: str) -> bool:
        """Check if URL is a YouTube Shorts video"""
        return '/shorts/' in url

class FileManager:
    """Utility class for file management"""
    
    @staticmethod
    def ensure_dir(path: Path) -> None:
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def cleanup_old_files(directory: Path, max_age_days: int = 7) -> None:
        """Clean up old files"""
        try:
            for file in directory.glob("*"):
                if file.is_file():
                    age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
                    if age.days > max_age_days:
                        file.unlink()
        except Exception as e:
            st.warning(f"Error cleaning up files: {str(e)}")
    
    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """Get total size of directory in bytes"""
        total = 0
        for entry in directory.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

class AudioUtility:
    """Audio file handling utilities"""
    
    # Supported audio formats and their max sizes in MB
    SUPPORTED_FORMATS = {
        '.mp3': {'max_size': 500, 'mime': 'audio/mpeg'},
        '.wav': {'max_size': 1000, 'mime': 'audio/wav'},
        '.m4a': {'max_size': 500, 'mime': 'audio/mp4'},
        '.ogg': {'max_size': 500, 'mime': 'audio/ogg'},
        '.flac': {'max_size': 1000, 'mime': 'audio/flac'},
        '.aac': {'max_size': 500, 'mime': 'audio/aac'},
        '.wma': {'max_size': 500, 'mime': 'audio/x-ms-wma'}
    }

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio file duration using ffmpeg"""
        try:
            probe = ffmpeg.probe(file_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception:
            return 0

    @staticmethod
    def convert_to_mp3(input_path: str, output_path: str) -> bool:
        """Convert audio file to MP3 format"""
        try:
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', q=2)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return True
        except Exception:
            return False

    @classmethod
    def validate_audio_file(cls, file_info: dict) -> dict:
        """
        Validate audio file
        Returns: dict with 'valid' boolean and 'message' string
        """
        extension = file_info['extension'].lower()
        
        # Check format
        if extension not in cls.SUPPORTED_FORMATS:
            return {
                'valid': False,
                'message': f"Unsupported format. Supported formats: {', '.join(cls.SUPPORTED_FORMATS.keys())}"
            }
        
        # Check size
        max_size = cls.SUPPORTED_FORMATS[extension]['max_size'] * 1024 * 1024  # Convert to bytes
        if file_info['size'] > max_size:
            return {
                'valid': False,
                'message': f"File too large. Maximum size for {extension} is {cls.SUPPORTED_FORMATS[extension]['max_size']}MB"
            }
        
        if file_info['size'] < 1024:  # 1KB minimum
            return {
                'valid': False,
                'message': "File too small to be valid audio"
            }
        
        return {
            'valid': True,
            'message': "File validation successful"
        }

def check_system_compatibility() -> tuple[bool, List[str]]:
    """Check system compatibility for transcription"""
    issues = []
    compatible = True
    

    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        compatible = False
        issues.append("Python 3.8 or higher required")
    
    # Check GPU
    if not torch.cuda.is_available():
        issues.append("No GPU detected - processing will be slower")
    else:
        gpu_info = torch.cuda.get_device_properties(0)
        gpu_mem = gpu_info.total_memory / (1024**3)
        gpu_name = gpu_info.name
        issues.append(f"GPU detected: {gpu_name} with {gpu_mem:.1f}GB VRAM")
        if gpu_mem < 4:
            issues.append(f"Low GPU memory ({gpu_mem:.1f}GB) - larger models may not work")
    
   # Check CUDA version
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        issues.append(f"CUDA Version: {cuda_version}")

    # Check FFmpeg
    if not shutil.which('ffmpeg'):
        compatible = False
        issues.append("FFmpeg not found")
    
    # Check disk space
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 10:
            issues.append(f"Low disk space: {free_gb}GB free")
    except:
        issues.append("Could not check disk space")
    
    return compatible, issues

class ErrorConfig:
    """Configuration for error handling and retries"""
    RETRY_DELAYS = {
        'download': [5, 10, 20, 30, 60],
        'transcribe': [10, 20, 30, 60, 120],
        'rate_limit': [60, 120, 240, 480, 960]
    }
    
    MAX_RETRIES = {
        'download': 5,
        'transcribe': 3,
        'rate_limit': 5
    }
    
    ERROR_CATEGORIES = {
        'network': ['HTTP Error', 'ConnectionError', 'Timeout'],
        'availability': ['Video unavailable', 'Sign in', 'copyright'],
        'resources': ['cuda', 'out of memory', 'GPU memory'],
        'rate_limit': ['429', 'Too Many Requests'],
        'processing': ['ffmpeg', 'audio processing', 'transcription failed']
    }

class ProcessingConfig:
    """Configuration for processing parameters"""
    BATCH_SIZE = 3
    MEMORY_THRESHOLD = 0.8
    PROGRESS_SAVE_INTERVAL = 60
    CLEANUP_THRESHOLD = 10
    MAX_TRANSCRIPT_SIZE = 10 * 1024 * 1024  # 10MB
