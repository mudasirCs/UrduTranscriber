# YouTube Urdu Video Transcriber

An application that downloads YouTube videos, transcribes Urdu audio to text, and generates well-formatted documents with clickable timestamps.

## Features

- 🎥 YouTube video transcription
- 📝 Urdu language support
- 🎯 Multiple Whisper model options (tiny to large)
- ⌚ Clickable timestamps linking back to video
- 📄 Word document generation
- 🎨 Clean web interface
- 💾 Recent transcriptions history

## Prerequisites

- Python 3.8 or higher
- FFmpeg
- Urdu font: Jameel Noori Nastaleeq

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd youtube-transcriber
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open browser at `http://localhost:8501`

3. Enter YouTube URL

4. Select Whisper model:
- Tiny: Fastest, least accurate
- Base: Good balance
- Small: Better accuracy
- Medium: High accuracy
- Large: Best accuracy, slowest

5. Click "Start Transcription"

6. Download generated Word document

## Project Structure
```
youtube_transcriber/
│
├── src/
│   ├── __init__.py
│   ├── transcriber.py    # Core transcription logic
│   ├── video_preview.py  # Video info and preview
│   ├── utils.py         # Helper functions
│   └── styles.py        # UI styling
│
├── static/
│   └── styles.css       # Custom CSS
│
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Output Format

The generated Word document includes:
1. Video information
2. Complete transcript
3. Timestamped transcript with clickable links
4. Proper Urdu font formatting

## Models and Performance

| Model | Size | Speed | Accuracy | RAM Required |
|-------|------|-------|----------|--------------|
| Tiny  | 39M  | 32x   | Basic    | 1GB         |
| Base  | 74M  | 16x   | Good     | 1GB         |
| Small | 244M | 8x    | Better   | 2GB         |
| Medium| 769M | 4x    | Great    | 4GB         |
| Large | 1.5G | 2x    | Best     | 8GB         |

## Troubleshooting

1. FFmpeg not found:
```bash
# Windows
Add FFmpeg to your PATH environment variable

# Linux
sudo apt install ffmpeg

# Mac
brew install ffmpeg
```

2. Font issues:
- Install Jameel Noori Nastaleeq font
- Restart application

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper
- Streamlit
- yt-dlp
- python-docx

## Support

For support, please open an issue in the GitHub repository.