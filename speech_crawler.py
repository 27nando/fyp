import os
import datetime
import statistics
import time
import re
import sys
from webvtt import WebVTT
import speech_recognition as sr
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, TranslationLanguageNotAvailable
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from googleapiclient.errors import HttpError

# Video duration constraints
MIN_VIDEO_DURATION = 60  # 1 minute
MAX_VIDEO_DURATION = 1200  # 20 minutes

def parse_duration(duration: str) -> int:
    """Convert YouTube duration format (PT1H2M10S) to seconds"""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return hours * 3600 + minutes * 60 + seconds

# Load environment variables
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

if not API_KEY:
    print("Error: YouTube API key not found!")
    print("Please make sure to:")
    print("1. Create a .env file in the project directory")
    print("2. Add your YouTube API key to the .env file as: YOUTUBE_API_KEY=your_api_key_here")
    sys.exit(1)

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# List of news channels to analyze - based on Harvard Media Bias Chart
# Selected channels with strong bias and reliable English subtitles
NEWS_CHANNELS = {
    # Left-leaning sources
    'MSNBC': 'UCaXkIU1QidjPwiAYu6GcHjg',        # Strong left bias, opinion-focused
    'Democracy Now': 'UCzuqE7-t13O4NIDYJfakrhw',  # Progressive left perspective
    'The Young Turks': 'UC1yBKRuGpC1tSM73A0ZjYjQ', # Progressive advocacy
    
    # Right-leaning sources
    'Fox News': 'UCXIJgqnII2ZOINSWNOGFThA',      # Strong right bias, opinion-focused
    'Newsmax': 'UCx6h-dWzJ5NpAlja1YsApdg',       # Conservative perspective
    'OAN': 'UCe02lGcO-ahAURWuxAJnjdA',          # One America News, far-right perspective
    
    # Extreme partisan sources
    'The Daily Wire': 'UCaeO5vkdj5xOQHp4UmIN6dw', # Strong conservative bias
    'BlazeTV': 'UCWA6ZkxA1jz9sN5TxPO9E5g'        # Strong conservative commentary
}

class SubtitleProcessor:
    """Process and clean subtitles using advanced filtering techniques"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.everything_cool = re.compile(r"^[A-Za-z0-9\,\.\-\?\"\'\'\!\"\s\;\:\"\"\–\'\'\'\/\\]+$", re.IGNORECASE)
        self.leave_chars = re.compile(r"[^a-z\s\']", re.IGNORECASE)
        self.html_tags = re.compile(r'<.*?>')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced validation and processing"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Initial text quality check
        initial_length = len(text)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / initial_length < 0.3:  # Less than 30% letters
            return ""
            
        # Remove HTML and XML tags
        text = self.html_tags.sub(' ', text)
        
        # Remove common transcript markers and noise indicators
        noise_markers = [
            # Music and sound effects
            '[music]', '[♪]', '[♫]', '[song]', '[singing]', '[humming]',
            # Audience reactions
            '[applause]', '[laughter]', '[cheering]', '[booing]', '[gasps]',
            # Background and technical
            '[background noise]', '[static]', '[interference]', '[echo]',
            # Speech qualifiers
            '[inaudible]', '[mumbling]', '[whispering]', '[speaking foreign language]',
            # Multiple speakers
            '[crosstalk]', '[overlapping voices]', '[multiple speakers]',
            # Silence and pauses
            '[silence]', '[pause]', '[long pause]', '[brief pause]',
            # Technical markers
            '[recording starts]', '[recording ends]', '[cut]', '[edit]'
        ]
        
        for marker in noise_markers:
            text = re.sub(re.escape(marker), '', text, flags=re.IGNORECASE)
        
        # Remove speaker labels and timestamps
        text = re.sub(r'\[?\d{1,2}:\d{2}(:\d{2})?\]?', '', text)  # [00:00] or [00:00:00]
        text = re.sub(r'\[?speaker\s*\d*\]?:?', '', text, flags=re.IGNORECASE)  # [Speaker 1]: or Speaker 2:
        
        # Normalize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes to regular quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to regular apostrophes
        
        # Normalize dashes and hyphens
        text = text.replace('–', '-').replace('—', '-')  # Em/en dashes to hyphens
        
        # Remove special characters but keep essential punctuation
        text = self.leave_chars.sub(' ', text)
        
        # Normalize whitespace and remove unwanted characters
        text = text.replace('\n', ' ')  # Newlines
        text = text.replace('\t', ' ')  # Tabs
        text = text.replace('\r', ' ')  # Carriage returns
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Final validation
        if len(text) < initial_length * 0.3:  # Lost too much content
            return ""
            
        words = text.split()
        if len(words) < 2:  # Too few words
            return ""
            
        # Check for repeated words (possible transcription error)
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0]
        if most_common[1] > len(words) * 0.5:  # Same word repeated too much
            return ""
            
        return text
    
    def filter_subtitles(self, subtitles: List[Dict[str, Any]], 
                        min_duration: float = 1.0,
                        max_duration: float = 15.0,
                        min_words: int = 5,
                        max_words: int = 50) -> List[Dict[str, Any]]:
        """Filter subtitles with enhanced validation and quality checks"""
        if not subtitles:
            return []
            
        filtered_subs = []
        total_duration = sum(sub.get('duration', 0) for sub in subtitles if isinstance(sub, dict))
        avg_duration = total_duration / len(subtitles) if subtitles else 0
        
        for i, sub in enumerate(subtitles):
            try:
                # Basic validation
                if not isinstance(sub, dict):
                    continue
                    
                # Validate required fields
                required_fields = ['text', 'start', 'duration']
                if not all(key in sub for key in required_fields):
                    continue
                    
                # Validate field types
                if not isinstance(sub['text'], str):
                    continue
                if not all(isinstance(sub[key], (int, float)) for key in ['start', 'duration']):
                    continue
                    
                # Get and validate text
                text = sub['text'].strip()
                if not text:
                    continue
                    
                # Check text length and word count
                words = text.split()
                word_count = len(words)
                if word_count < min_words or word_count > max_words:
                    continue
                    
                # Check duration
                duration = float(sub['duration'])
                if duration < min_duration or duration > max_duration:
                    continue
                    
                # Check if duration is abnormal compared to average
                if duration > avg_duration * 3:  # Suspiciously long
                    continue
                    
                # Check speech rate (words per minute)
                speech_rate = (word_count / duration) * 60
                if speech_rate > 200:  # Too fast to be natural speech
                    continue
                    
                # Check text quality
                alpha_ratio = sum(c.isalpha() for c in text) / len(text)
                if alpha_ratio < 0.5:  # At least 50% should be letters
                    continue
                    
                # Check for repetitive words
                word_counts = Counter(words)
                most_common = word_counts.most_common(1)[0]
                if most_common[1] > len(words) * 0.5:  # Same word repeated too much
                    continue
                    
                # Clean and validate text
                cleaned_text = self.clean_text(text)
                if not cleaned_text:
                    continue
                    
                # Calculate additional metrics
                cleaned_words = cleaned_text.split()
                cleaned_word_count = len(cleaned_words)
                content_retention = len(cleaned_text) / len(text)
                
                # Skip if too much content was lost in cleaning
                if content_retention < 0.5:
                    continue
                    
                # Skip if cleaned text is too short
                if cleaned_word_count < min_words:
                    continue
                    
                # Create enhanced subtitle entry
                filtered_sub = {
                    'start': float(sub['start']),
                    'duration': duration,
                    'end': float(sub['start']) + duration,
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'word_count': cleaned_word_count,
                    'speech_rate': speech_rate,
                    'alpha_ratio': alpha_ratio,
                    'content_retention': content_retention
                }
                
                filtered_subs.append(filtered_sub)
                
            except Exception as e:
                print(f"Error processing subtitle {i}: {str(e)}")
                continue
            
        return filtered_subs
    
    def merge_subtitles(self, subtitles: List[Dict[str, Any]], 
                       max_gap: float = 1.5,
                       max_merged_duration: float = 10.0,
                       min_merged_words: int = 10,
                       max_merged_words: int = 100) -> List[Dict[str, Any]]:
        """Merge consecutive subtitles with enhanced validation and processing"""
        if not subtitles:
            return []
            
        # Validate input subtitles
        valid_subs = []
        for sub in subtitles:
            if not isinstance(sub, dict):
                continue
                
            required_keys = ['start', 'end', 'duration', 'text']
            if not all(key in sub for key in required_keys):
                continue
                
            if not all(isinstance(sub[key], (int, float)) for key in ['start', 'end', 'duration']):
                continue
                
            if not isinstance(sub['text'], str):
                continue
                
            valid_subs.append(sub)
            
        if not valid_subs:
            return []
        
        merged = []
        current = valid_subs[0].copy()
        
        for next_sub in valid_subs[1:]:
            # Calculate timing metrics
            gap = next_sub['start'] - (current['end'] if 'end' in current else current['start'] + current['duration'])
            merged_duration = next_sub['end'] - current['start']
            
            # Calculate text metrics
            merged_text = current['text'] + ' ' + next_sub['text']
            word_count = len(merged_text.split())
            
            # Check if we should merge
            if (gap <= max_gap and 
                merged_duration <= max_merged_duration and 
                word_count >= min_merged_words and 
                word_count <= max_merged_words):
                # Merge subtitles
                current['end'] = next_sub['end']
                current['duration'] = current['end'] - current['start']
                current['text'] = merged_text
                current['cleaned_text'] = self.clean_text(merged_text)
                current['word_count'] = word_count
            else:
                # Clean up current subtitle before adding
                if 'cleaned_text' not in current:
                    current['cleaned_text'] = self.clean_text(current['text'])
                if 'word_count' not in current:
                    current['word_count'] = len(current['text'].split())
                    
                merged.append(current)
                current = next_sub.copy()
        
        # Add the last subtitle
        if 'cleaned_text' not in current:
            current['cleaned_text'] = self.clean_text(current['text'])
        if 'word_count' not in current:
            current['word_count'] = len(current['text'].split())
            
        merged.append(current)
        
        # Final validation of merged subtitles
        final_merged = []
        for sub in merged:
            # Skip if merged subtitle is too short or too long
            if not (min_merged_words <= sub['word_count'] <= max_merged_words):
                continue
                
            # Skip if duration is invalid
            if not (0 < sub['duration'] <= max_merged_duration):
                continue
                
            final_merged.append(sub)
            
        return final_merged

class SpeechCrawler:
    """Enhanced crawler for extracting and analyzing speech from YouTube videos"""
    
    def __init__(self):
        self.subtitle_processor = SubtitleProcessor()
        
    def verify_api_key(self) -> bool:
        """Verify that the API key works and handle expiration"""
        try:
            youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
            request = youtube.channels().list(
                part="snippet",
                id="UCXIJgqnII2ZOINSWNOGFThA"
            )
            request.execute()
            return True
        except HttpError as e:
            error_details = str(e)
            if "API key expired" in error_details:
                print("\nError: Your YouTube API key has expired!")
                print("Please follow these steps to fix:")
                print("1. Go to Google Cloud Console: https://console.cloud.google.com/apis/credentials")
                print("2. Find your API key and click on it")
                print("3. Click 'REGENERATE KEY' to get a new key")
                print("4. Update your .env file with the new key")
                print("\nNote: After regenerating, it may take a few minutes for the new key to become active.")
                return False
            elif "quota" in error_details.lower():
                print("\nError: YouTube API quota exceeded!")
                print("Please wait until your quota resets or:")
                print("1. Go to Google Cloud Console: https://console.cloud.google.com/apis/dashboard")
                print("2. Check your current quota usage and limits")
                print("3. Consider requesting a quota increase if needed")
                return False
            elif "API key not valid" in error_details:
                print("\nError: Invalid YouTube API key!")
                print("Please make sure to:")
                print("1. Create a new project in Google Cloud Console")
                print("2. Enable the YouTube Data API v3")
                print("3. Create an API key")
                print("4. Add the API key to your .env file")
                print("\nDetailed error:", str(e))
                return False
            else:
                print(f"\nUnexpected API error: {error_details}")
                return False
        except Exception as e:
            print("\nUnexpected error:", str(e))
            return False
    
    def get_video_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Get and process transcript for a video with enhanced validation and cleaning"""
        def validate_entries(transcript_list) -> List[Dict[str, Any]]:
            valid_entries = []
            for entry in transcript_list:
                if not isinstance(entry, dict):
                    continue
                    
                # Validate required fields
                required_fields = ['text', 'start', 'duration']
                if not all(key in entry for key in required_fields):
                    continue
                    
                # Validate field types
                if not isinstance(entry['text'], str):
                    continue
                if not all(isinstance(entry[key], (int, float)) for key in ['start', 'duration']):
                    continue
                    
                # Validate field values
                if entry['duration'] <= 0:
                    continue
                if not entry['text'].strip():
                    continue
                    
                # Clean and validate text
                text = str(entry['text']).strip()
                text = ' '.join(text.split())  # Normalize whitespace
                text = text.replace('\n', ' ')  # Remove newlines
                
                # Skip entries with too many non-alphanumeric characters
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text) if text else 0
                if alpha_ratio < 0.5:  # At least 50% should be letters or spaces
                    continue
                    
                # Skip entries that are likely music or sound effects
                lower_text = text.lower()
                if any(marker in lower_text for marker in ['♪', '♫', '[music]', '[applause]', '[laughter]']):
                    continue
                    
                # Create validated entry
                valid_entries.append({
                    'text': text,
                    'start': float(entry['start']),
                    'duration': float(entry['duration']),
                    'end': float(entry['start']) + float(entry['duration']),
                    'word_count': len(text.split()),
                    'alpha_ratio': alpha_ratio
                })
            return valid_entries

        try:
            transcript_entries = []
            error_msg = ""
            
            # Try manual captions first
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_entries = validate_entries(transcript_list)
                if transcript_entries:
                    print(f"Using manual captions for video {video_id}")
                    return transcript_entries
            except TranscriptsDisabled:
                error_msg = "Manual captions are disabled"
            except NoTranscriptFound:
                error_msg = "No manual captions found"
            except Exception as e:
                error_msg = f"Error getting manual captions: {str(e)}"

            # Try auto-generated captions
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en'])
                transcript_entries = validate_entries(transcript_list)
                if transcript_entries:
                    print(f"Using auto-generated captions for video {video_id}")
                    return transcript_entries
            except TranscriptsDisabled:
                error_msg += ", Auto-captions are disabled"
            except NoTranscriptFound:
                error_msg += ", No auto-captions found"
            except Exception as e:
                error_msg += f", Error getting auto-captions: {str(e)}"

            print(f"No valid transcript available for video {video_id}: {error_msg}")
            return []
            
        except VideoUnavailable as e:
            print(f"Video {video_id} is unavailable: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error processing video {video_id}: {str(e)}")
            return []

    def process_video_subtitles(self, video_id: str) -> Dict[str, Any]:
        """Process subtitles for a video with enhanced validation"""
        transcript_entries = self.get_video_transcript(video_id)
        
        if not transcript_entries:
            return {}
            
        try:
            # Combine transcript entries into a single text
            full_text = ' '.join(entry['text'] for entry in transcript_entries)
            
            # Basic text cleaning
            full_text = re.sub(r'\s+', ' ', full_text)  # Remove extra whitespace
            full_text = full_text.strip()
            
            if not full_text:  # If text is empty after cleaning
                return {}
                
            # Calculate metrics
            total_duration = sum(entry['duration'] for entry in transcript_entries)
            word_count = len(full_text.split())
            
            if total_duration == 0:  # Avoid division by zero
                speech_rate = 0
            else:
                speech_rate = word_count / (total_duration / 60)  # Words per minute
                
            return {
                'transcript': full_text,
                'duration_seconds': total_duration,
                'word_count': word_count,
                'speech_rate': speech_rate,
                'segment_count': len(transcript_entries)
            }
            
        except Exception as e:
            print(f"Error processing subtitles for video {video_id}: {str(e)}")
            return {}

    def has_english_subtitles(self, video_id: str) -> bool:
        """Check if a video has valid English subtitles with enhanced validation"""
        try:
            # Try manual captions first
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Check for manual English captions
                if transcript_list.find_manually_created_transcript(['en']):
                    print(f"Found manual English captions for video {video_id}")
                    return True
                    
                # Check for manual captions that can be translated to English
                manual_transcripts = transcript_list.find_manually_created_transcript()
                if manual_transcripts and manual_transcripts.is_translatable:
                    print(f"Found translatable manual captions for video {video_id}")
                    return True
                    
            except (NoTranscriptFound, TranscriptsDisabled):
                pass  # Try auto-generated captions
                
            # Try auto-generated captions
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Check for auto-generated English captions
                if transcript_list.find_generated_transcript(['en-US', 'en']):
                    print(f"Found auto-generated English captions for video {video_id}")
                    return True
                    
                # Check for auto-generated captions that can be translated
                auto_transcripts = transcript_list.find_generated_transcript()
                if auto_transcripts and auto_transcripts.is_translatable:
                    print(f"Found translatable auto-generated captions for video {video_id}")
                    return True
                    
            except (NoTranscriptFound, TranscriptsDisabled) as e:
                print(f"No captions available for video {video_id}: {str(e)}")
                return False
                
            return False
            
        except VideoUnavailable as e:
            print(f"Video {video_id} is unavailable: {str(e)}")
            return False
        except Exception as e:
            print(f"Error checking subtitles for video {video_id}: {str(e)}")
            return False

    def get_channel_videos(self, channel_id: str, max_results: int = 100) -> pd.DataFrame:
        """Get videos from a specific channel with enhanced metadata, rate limiting, and exponential backoff"""
        def exponential_backoff(retry_count: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
            """Calculate delay with exponential backoff"""
            delay = min(base_delay * (2 ** retry_count), max_delay)
            return delay
            
        try:
            youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
            videos = []
            page_token = None
            retry_count = 0
            max_retries = 3
            
            while len(videos) < max_results and retry_count < max_retries:
                try:
                    # Add delay between API calls with exponential backoff
                    delay = exponential_backoff(retry_count)
                    print(f"Waiting {delay:.1f}s before next API call...")
                    time.sleep(delay)
                    
                    # Search for videos with captions (reduced batch size)
                    search_request = youtube.search().list(
                        channelId=channel_id,
                        part='id,snippet',
                        order='date',
                        type='video',
                        videoCaption='closedCaption',  # Only fetch videos with captions
                        maxResults=min(max_results - len(videos), 25),  # Reduced from 50
                        pageToken=page_token
                    )
                    
                    search_response = search_request.execute()
                    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
                    
                    if not video_ids:
                        break
                    
                    # Process videos in smaller batches
                    for i in range(0, len(video_ids), 10):  # Reduced from 50
                        batch_ids = video_ids[i:i+10]
                        
                        # Add delay between API calls
                        delay = exponential_backoff(retry_count)
                        print(f"Waiting {delay:.1f}s before fetching video details...")
                        time.sleep(delay)
                        
                        try:
                            # Get detailed video information
                            video_request = youtube.videos().list(
                                part='contentDetails,statistics,snippet',
                                id=','.join(batch_ids)
                            )
                            video_response = video_request.execute()
                            
                            for video in video_response.get('items', []):
                                video_id = video['id']
                                
                                # Add delay before checking subtitles
                                time.sleep(1)
                                
                                # Check for English subtitles
                                if not self.has_english_subtitles(video_id):
                                    continue
                                
                                # Get duration in seconds
                                duration = parse_duration(video['contentDetails']['duration'])
                                
                                # Skip videos that are too short or too long
                                if duration < MIN_VIDEO_DURATION or duration > MAX_VIDEO_DURATION:
                                    continue
                                
                                # Create video data entry
                                video_data = {
                                    'video_id': video_id,
                                    'title': video['snippet']['title'],
                                    'channel_title': video['snippet']['channelTitle'],
                                    'published_at': video['snippet']['publishedAt'],
                                    'description': video['snippet']['description'],
                                    'duration': duration,
                                    'view_count': int(video['statistics'].get('viewCount', 0)),
                                    'like_count': int(video['statistics'].get('likeCount', 0)),
                                    'comment_count': int(video['statistics'].get('commentCount', 0))
                                }
                                videos.append(video_data)
                                
                                if len(videos) >= max_results:
                                    break
                                    
                        except HttpError as e:
                            if 'quotaExceeded' in str(e):
                                print(f"\nYouTube API quota exceeded. Saving {len(videos)} videos collected so far.")
                                return pd.DataFrame(videos)
                            print(f"Error getting video details: {str(e)}")
                            retry_count += 1
                            continue
                        
                        if len(videos) >= max_results:
                            break
                    
                    # Reset retry count on successful iteration
                    retry_count = 0
                    
                    # Get next page token
                    page_token = search_response.get('nextPageToken')
                    if not page_token or len(videos) >= max_results:
                        break
                        
                except HttpError as e:
                    if 'quotaExceeded' in str(e):
                        print(f"\nYouTube API quota exceeded. Saving {len(videos)} videos collected so far.")
                        return pd.DataFrame(videos)
                    print(f"Error searching videos: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
            
            if not videos:
                print(f"No videos with English subtitles found for channel {channel_id}")
                return pd.DataFrame()
            
            print(f"Found {len(videos)} videos with English subtitles")
            return pd.DataFrame(videos)
            
        except Exception as e:
            print(f"\nError fetching videos for channel {channel_id}: {str(e)}")
            return pd.DataFrame(videos) if videos else pd.DataFrame()
    
    def get_video_transcript(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Extract and process transcript for a given video ID"""
        try:
            # Get raw transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Process subtitles
            processed_subs = self.subtitle_processor.filter_subtitles(transcript_list)
            merged_subs = self.subtitle_processor.merge_subtitles(processed_subs)
            
            return merged_subs
        except Exception as e:
            print(f"Could not get transcript for video {video_id}: {str(e)}")
            return None
    
    def analyze_speech_patterns(self, transcript_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze speech patterns in the transcript entries with enhanced metrics"""
        try:
            if not transcript_entries:
                return {}
                
            # Initialize metrics
            total_duration = 0.0
            total_words = 0
            total_segments = len(transcript_entries)
            segment_durations = []
            words_per_segment = []
            speech_rates = []
            pauses = []
            
            # Process each segment
            for i, entry in enumerate(transcript_entries):
                # Basic validation
                if not isinstance(entry, dict) or 'text' not in entry or 'duration' not in entry:
                    continue
                    
                # Get text and duration
                text = entry['text'].strip()
                duration = float(entry['duration'])
                
                if not text or duration <= 0:
                    continue
                    
                # Calculate segment metrics
                words = len(text.split())
                speech_rate = (words / duration) * 60  # Words per minute
                
                # Store metrics
                total_duration += duration
                total_words += words
                segment_durations.append(duration)
                words_per_segment.append(words)
                speech_rates.append(speech_rate)
                
                # Calculate pause between segments
                if i > 0:
                    prev_end = transcript_entries[i-1].get('start', 0) + transcript_entries[i-1].get('duration', 0)
                    curr_start = entry.get('start', 0)
                    pause = max(0, curr_start - prev_end)
                    pauses.append(pause)
            
            # Calculate aggregate metrics
            avg_duration = sum(segment_durations) / len(segment_durations) if segment_durations else 0
            avg_words = sum(words_per_segment) / len(words_per_segment) if words_per_segment else 0
            avg_speech_rate = sum(speech_rates) / len(speech_rates) if speech_rates else 0
            avg_pause = sum(pauses) / len(pauses) if pauses else 0
            
            # Calculate variability metrics
            duration_std = statistics.stdev(segment_durations) if len(segment_durations) > 1 else 0
            words_std = statistics.stdev(words_per_segment) if len(words_per_segment) > 1 else 0
            speech_rate_std = statistics.stdev(speech_rates) if len(speech_rates) > 1 else 0
            pause_std = statistics.stdev(pauses) if len(pauses) > 1 else 0
            
            return {
                'total_metrics': {
                    'duration': total_duration,
                    'words': total_words,
                    'segments': total_segments,
                    'overall_speech_rate': (total_words / total_duration * 60) if total_duration > 0 else 0
                },
                'segment_metrics': {
                    'avg_duration': avg_duration,
                    'duration_std': duration_std,
                    'avg_words': avg_words,
                    'words_std': words_std,
                    'avg_speech_rate': avg_speech_rate,
                    'speech_rate_std': speech_rate_std
                },
                'pause_metrics': {
                    'avg_pause': avg_pause,
                    'pause_std': pause_std,
                    'total_pause_time': sum(pauses) if pauses else 0
                }
            }
            
        except Exception as e:
            print(f"Error analyzing speech patterns: {str(e)}")
            return {}
            
            for entry in transcript_entries:
                cleaned_text = self.subtitle_processor.clean_text(entry['text'])
                if cleaned_text:
                    cleaned_texts.append(cleaned_text)
                    total_duration += entry['duration']
            
            if not cleaned_texts:
                return {}
                
            # Combine all cleaned text
            full_text = ' '.join(cleaned_texts)
            words = full_text.split()
            
            # Calculate metrics
            total_words = len(words)
            unique_words = len(set(words))
            
            # Avoid division by zero
            vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
            speech_rate = (total_words / (total_duration / 60)) if total_duration > 0 else 0
            
            return {
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_diversity': vocabulary_diversity,
                'speech_rate': speech_rate,
                'cleaned_texts': cleaned_texts  # Include cleaned texts for later use
            }
        except Exception as e:
            print(f"Error analyzing speech patterns: {str(e)}")
            return {}

    def crawl_news_channels(self, max_videos_per_channel: int = 100):
        """Crawl videos from specified news channels and extract their transcripts"""
        all_data = []
        
        for channel_name, channel_id in NEWS_CHANNELS.items():
            print(f"\nCrawling videos from {channel_name}...")
            channel_data = []
            
            # Get videos from channel
            videos_df = self.get_channel_videos(channel_id, max_results=max_videos_per_channel)
            
            if len(videos_df) == 0:
                print(f"No videos found for {channel_name}, skipping...")
                continue
                
            print(f"Found {len(videos_df)} videos, extracting transcripts...")
            
            # Process each video
            for _, video in videos_df.iterrows():
                video_id = video['video_id']
                try:
                    # Get and process transcript
                    transcript_entries = self.get_video_transcript(video_id)
                    if not transcript_entries:
                        continue
                        
                    # Process subtitles
                    processed_subs = self.subtitle_processor.filter_subtitles(
                        transcript_entries,
                        min_duration=0.5,
                        max_duration=10.0,
                        min_words=3,
                        max_words=30
                    )
                    
                    if not processed_subs:
                        continue
                        
                    merged_subs = self.subtitle_processor.merge_subtitles(
                        processed_subs,
                        max_gap=1.0,
                        max_merged_duration=8.0,
                        min_merged_words=5,
                        max_merged_words=50
                    )
                    
                    if not merged_subs:
                        continue
                        
                    # Analyze speech patterns
                    speech_metrics = self.analyze_speech_patterns(merged_subs)
                    if not speech_metrics:
                        continue
                        
                    # Create video data entry
                    video_data = {
                        'video_id': video_id,
                        'title': video['title'],
                        'channel_name': channel_name,
                        'published_at': video['published_at'],
                        'duration': video['duration'],
                        'view_count': video['view_count'],
                        'like_count': video['like_count'],
                        'comment_count': video['comment_count'],
                        'transcript': ' '.join(sub['cleaned_text'] for sub in merged_subs),
                        'segment_count': len(merged_subs),
                        **speech_metrics['total_metrics'],
                        **speech_metrics['segment_metrics'],
                        **speech_metrics['pause_metrics']
                    }
                    
                    channel_data.append(video_data)
                    print(f"Successfully processed video {video_id}")
                    
                except Exception as e:
                    print(f"Error processing video {video_id}: {str(e)}")
                    continue
            
            if channel_data:
                # Create DataFrame with channel data
                channel_df = pd.DataFrame(channel_data)
                all_data.append(channel_df)
                
                # Save channel data
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                channel_file = f"data/{channel_name.lower().replace(' ', '_')}_{timestamp}.csv"
                os.makedirs('data', exist_ok=True)
                channel_df.to_csv(channel_file, index=False)
                print(f"\nSaved {len(channel_data)} videos from {channel_name} to {channel_file}")
            
        if not all_data:
            print("\nNo data was collected. Please check the errors above.")
            return
            
        # Combine and save all data
        final_df = pd.concat(all_data, ignore_index=True)
        final_file = f"data/all_channels_{timestamp}.csv"
        final_df.to_csv(final_file, index=False)
        print(f"\nSaved {len(final_df)} videos from all channels to {final_file}")

def main():
    print("Starting Enhanced YouTube News Channel Crawler...")
    
    # Verify environment setup
    if not os.path.exists('.env'):
        print("\nError: .env file not found!")
        print("Please create a .env file with your YouTube API key:")
        print("YOUTUBE_API_KEY=your_api_key_here")
        return
        
    if not API_KEY:
        print("\nError: YouTube API key not found in .env file!")
        print("Please add your API key to the .env file:")
        print("YOUTUBE_API_KEY=your_api_key_here")
        return
    
    # Initialize crawler
    crawler = SpeechCrawler()
    
    # Verify API key before starting
    if not crawler.verify_api_key():
        return  # Error message already printed by verify_api_key()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    try:
        crawler.crawl_news_channels(max_videos_per_channel=100)
    except HttpError as e:
        error_details = str(e)
        if 'API key expired' in error_details:
            print("\nError: Your YouTube API key expired during crawling!")
            print("Please regenerate your API key and try again.")
        elif 'quota' in error_details.lower():
            print("\nError: YouTube API quota exceeded during crawling!")
            print("Please wait until your quota resets or request a quota increase.")
        else:
            print(f"\nYouTube API error during crawling: {error_details}")
    except Exception as e:
        print(f"\nUnexpected error during crawling: {str(e)}")

if __name__ == "__main__":
    main()
