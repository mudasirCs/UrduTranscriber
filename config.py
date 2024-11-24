from pathlib import Path
from typing import Dict, Any
import json

class AppConfig:
    DEFAULT_CONFIG = {
        'temp_dir': 'temp',
        'output_dir': 'transcripts',
        'cache_dir': 'cache',
        'max_retries': 3,
        'retry_delay': 5,
        'cleanup_days': 7,
        'max_concurrent_downloads': 2,
        'max_file_age_days': 30,
        'supported_languages': ['ur', 'en'],
        'default_language': 'ur',
        'font_name': 'Jameel Noori Nastaleeq',
        'font_size': 14,
        'rate_limit': {
            'enabled': True,
            'requests_per_minute': 30,
            'min_interval': 2
        },
        'download_options': {
            'format': 'bestaudio/best',
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'audio_quality': '192K'
        }
    }

    def __init__(self):
        self.config_file = Path('config.json')
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**self.DEFAULT_CONFIG, **config}
            except Exception:
                return self.DEFAULT_CONFIG
        else:
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
        self.save_config(self.config)

    def get_dirs(self) -> Dict[str, Path]:
        """Get directory paths"""
        return {
            'temp': Path(self.config['temp_dir']).absolute(),
            'output': Path(self.config['output_dir']).absolute(),
            'cache': Path(self.config['cache_dir']).absolute()
        }

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist"""
        for dir_path in self.get_dirs().values():
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def download_options(self) -> Dict[str, Any]:
        """Get download options for yt-dlp"""
        return self.config['download_options']

    @property
    def rate_limit(self) -> Dict[str, Any]:
        """Get rate limit settings"""
        return self.config['rate_limit']