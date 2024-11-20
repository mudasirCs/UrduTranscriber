import streamlit as st
import whisper
import yt_dlp
import os
from datetime import datetime
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import docx.oxml.shared
from docx.opc.constants import RELATIONSHIP_TYPE
from pathlib import Path
from .utils import format_duration, sanitize_filename, get_timestamp

class TranscriptionManager:
    def __init__(self):
        self.dirs = {
            'temp': Path("temp"),
            'output': Path("transcripts")
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

    @staticmethod
    def add_hyperlink(paragraph, text, url):
        """Add a hyperlink to a paragraph"""
        part = paragraph.part
        r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        
        hyperlink = docx.oxml.shared.OxmlElement('w:hyperlink')
        hyperlink.set(docx.oxml.shared.qn('r:id'), r_id)
        
        new_run = docx.oxml.shared.OxmlElement('w:r')
        rPr = docx.oxml.shared.OxmlElement('w:rPr')
        
        rStyle = docx.oxml.shared.OxmlElement('w:rStyle')
        rStyle.set(docx.oxml.shared.qn('w:val'), 'Hyperlink')
        rPr.append(rStyle)
        
        new_run.append(rPr)
        new_run.text = text
        hyperlink.append(new_run)
        paragraph._p.append(hyperlink)

    def download_audio(self, video_url):
        """Download audio from YouTube video"""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.dirs['temp'] / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_audio': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192'
                }]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                audio_path = self.dirs['temp'] / f"{info['id']}.mp3"
                return {
                    'path': str(audio_path),
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'channel': info.get('channel', 'Unknown Channel'),
                    'upload_date': info.get('upload_date', 'Unknown Date')
                }
        except Exception as e:
            raise Exception(f"Error downloading audio: {str(e)}")

    @staticmethod
    @st.cache_resource
    def load_model(model_name):
        """Load and cache Whisper model"""
        return whisper.load_model(model_name)

    def process_video(self, url, model_name="base", progress_callback=None):
        """Process video and generate transcript"""
        audio_info = None
        try:
            if progress_callback:
                progress_callback(0.1, "Loading model...")
            model = TranscriptionManager.load_model(model_name)  # Changed this line
            
            if progress_callback:
                progress_callback(0.2, "Downloading audio...")
            audio_info = self.download_audio(url)
            
            if progress_callback:
                progress_callback(0.4, "Transcribing...")
            result = model.transcribe(
                audio_info['path'],
                language="ur",
                task="transcribe",
                fp16=False
            )
            
            if progress_callback:
                progress_callback(0.8, "Generating document...")
            return self.save_transcript(result, audio_info, url)
            
        finally:
            if audio_info and os.path.exists(audio_info['path']):
                try:
                    os.remove(audio_info['path'])
                except:
                    pass

    def save_transcript(self, result, info, url):
        """Save transcript to Word document"""
        doc = Document()
        
        # Title
        title = doc.add_heading(info['title'], 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Video Information
        doc.add_heading('Video Information', level=1)
        p = doc.add_paragraph('Video URL: ')
        self.add_hyperlink(p, url, url)
        
        # Info table
        info_table = doc.add_table(rows=2, cols=2)
        info_table.style = 'Table Grid'
        info_table.rows[0].cells[0].text = 'Channel'
        info_table.rows[0].cells[1].text = info['channel']
        info_table.rows[1].cells[0].text = 'Duration'
        info_table.rows[1].cells[1].text = format_duration(info['duration'])
        
        # Complete transcript
        doc.add_heading('Complete Transcript', level=1)
        p = doc.add_paragraph()
        text_run = p.add_run(result['text'])
        text_run.font.name = 'Jameel Noori Nastaleeq'
        text_run.font.size = Pt(14)
        
        # Add page break
        doc.add_page_break()
        
        # Timestamped transcript
        doc.add_heading('Timestamped Transcript', level=1)
        
        # Extract video ID
        video_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else url.split('/')[-1]
        
        for segment in result['segments']:
            p = doc.add_paragraph()
            timestamp_seconds = int(segment['start'])
            timestamp_url = f"https://youtube.com/watch?v={video_id}&t={timestamp_seconds}"
            self.add_hyperlink(p, f"[{format_duration(segment['start'])} - {format_duration(segment['end'])}]", timestamp_url)
            
            text_run = p.add_run('\n' + segment['text'].strip())
            text_run.font.name = 'Jameel Noori Nastaleeq'
            text_run.font.size = Pt(14)
        
        # Save document
        timestamp = get_timestamp()
        safe_title = sanitize_filename(info['title'])
        doc_path = self.dirs['output'] / f"{timestamp}_{safe_title}.docx"
        doc.save(str(doc_path))
        
        return str(doc_path)

    def list_recent_transcripts(self, limit=5):
        """Get list of recent transcriptions"""
        try:
            files = sorted(
                [f for f in self.dirs['output'].glob('*.docx')],
                key=os.path.getmtime,
                reverse=True
            )[:limit]
            return files
        except Exception:
            return []