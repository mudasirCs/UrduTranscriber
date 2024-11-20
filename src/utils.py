from datetime import datetime
import os

def format_duration(seconds):
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

def format_size(size_bytes):
    """Convert bytes to human-readable size"""
    if not size_bytes:
        return "0B"
        
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def sanitize_filename(title):
    """Create safe filename from title"""
    return "".join(x for x in title if x.isalnum() or x in (' ', '-', '_'))[:50]

def get_timestamp():
    """Get current timestamp for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class ModelConfig:
    MODELS = {
        "Tiny (fastest)": {
            "key": "tiny",
            "size": "39M parameters",
            "description": "Fastest but least accurate",
            "download_size": "~75 MB",
            "speed_factor": 0.5
        },
        "Base": {
            "key": "base",
            "size": "74M parameters",
            "description": "Good balance of speed and accuracy",
            "download_size": "~142 MB",
            "speed_factor": 1.0
        },
        "Small": {
            "key": "small",
            "size": "244M parameters",
            "description": "Better accuracy, still reasonable speed",
            "download_size": "~466 MB",
            "speed_factor": 2.0
        },
        "Medium": {
            "key": "medium",
            "size": "769M parameters",
            "description": "High accuracy, slower processing",
            "download_size": "~1.5 GB",
            "speed_factor": 3.0
        },
        "Large": {
            "key": "large",
            "size": "1550M parameters",
            "description": "Best accuracy, slowest processing",
            "download_size": "~3 GB",
            "speed_factor": 4.0
        }
    }