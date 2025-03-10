from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import sys
from googleapiclient.errors import HttpError

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

# List of news channels to analyze - focused on channels with reliable transcripts
NEWS_CHANNELS = {
    'VOA News': 'UCVSNOxehfALut52NbkfRBaA',    # Voice of America - clear English
    'DW News': 'UCknLrEdhRCp1aegoMqRaCZg',      # Deutsche Welle - English service
    'Global News': 'UChLtXXpo4Ge1ReTEboVvTDg',   # Canadian news
    'FRANCE 24 English': 'UCQfwfsi5VrQ8yKZ-UWmAEFg',  # International news
    'Al Jazeera English': 'UCNye-wNBqNL5ZzHSJj3l8Bg'  # High quality production
}

# Minimum video duration in seconds (skip very short clips)
MIN_VIDEO_DURATION = 60
# Maximum video duration in seconds (skip very long videos)
MAX_VIDEO_DURATION = 1200  # 20 minutes
# Minimum transcript word count
MIN_TRANSCRIPT_WORDS = 50

def verify_api_key():
    """Verify that the API key works"""
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        # Make a simple request to verify the key
        request = youtube.channels().list(
            part="snippet",
            id="UCXIJgqnII2ZOINSWNOGFThA"
        )
        request.execute()
        return True
    except HttpError as e:
        if "API key not valid" in str(e):
            print("\nError: Invalid YouTube API key!")
            print("Please make sure to:")
            print("1. Create a new project in Google Cloud Console")
            print("2. Enable the YouTube Data API v3")
            print("3. Create an API key")
            print("4. Add the API key to your .env file")
            print("\nDetailed error:", str(e))
        else:
            print("\nAPI Error:", str(e))
        return False
    except Exception as e:
        print("\nUnexpected error:", str(e))
        return False

def parse_duration(duration):
    """Convert YouTube duration format (PT1H2M10S) to seconds"""
    import re
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return hours * 3600 + minutes * 60 + seconds

def get_channel_videos(channel_id, max_results=100):
    """Get videos from a specific channel with enhanced metadata and filtering"""
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        
        # First search for videos with captions
        search_response = youtube.search().list(
            channelId=channel_id,
            part='id,snippet',
            order='date',
            type='video',
            videoCaption='closedCaption',  # Only get videos with captions
            maxResults=min(max_results * 2, 50)  # Get more since we'll filter some out
        ).execute()

        videos = []
        processed = set()
        
        while len(videos) < max_results:
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                break
                
            # Get detailed video information
            video_response = youtube.videos().list(
                part='contentDetails,statistics,snippet',
                id=','.join(video_ids)
            ).execute()
            
            for video in video_response.get('items', []):
                video_id = video['id']
                
                if video_id in processed:
                    continue
                    
                processed.add(video_id)
                
                # Get duration in seconds
                duration = parse_duration(video['contentDetails']['duration'])
                
                # Skip videos that are too short or too long
                if duration < MIN_VIDEO_DURATION or duration > MAX_VIDEO_DURATION:
                    continue
                    
                # Skip videos with disabled comments (might indicate lower quality)
                if video['contentDetails'].get('license') == 'youtube':
                    continue
                    
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
                    
            if 'nextPageToken' in search_response and len(videos) < max_results:
                search_response = youtube.search().list(
                    channelId=channel_id,
                    part='id,snippet',
                    order='date',
                    type='video',
                    videoCaption='closedCaption',
                    maxResults=50,
                    pageToken=search_response['nextPageToken']
                ).execute()
            else:
                break
        
        return pd.DataFrame(videos)
    except HttpError as e:
        print(f"\nError fetching videos for channel {channel_id}:", str(e))
        return pd.DataFrame()

def get_video_transcript(video_id):
    """Extract and process transcript for a given video ID with enhanced error handling"""
    try:
        # Try manual captions first
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            # Fall back to auto-generated captions
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en'])
            except Exception as e:
                print(f"No transcript available for video {video_id}: {str(e)}")
                return None
        
        # Process and clean transcript entries
        cleaned_entries = []
        for entry in transcript_list:
            if not isinstance(entry, dict):
                continue
                
            if 'text' not in entry:
                continue
                
            text = str(entry['text']).strip()
            if not text:
                continue
                
            # Basic cleaning
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.replace('\n', ' ')  # Remove newlines
            text = text.replace('[Music]', '')  # Remove common markers
            text = text.replace('[Applause]', '')
            text = text.replace('[Laughter]', '')
            text = text.replace('[Background Noise]', '')
            text = text.replace('[Inaudible]', '')
            
            cleaned_entries.append(text)
        
        if not cleaned_entries:
            return None
            
        # Combine all entries
        full_transcript = ' '.join(cleaned_entries)
        
        # Skip if transcript is too short
        if len(full_transcript.split()) < MIN_TRANSCRIPT_WORDS:
            return None
            
        return full_transcript
        
    except Exception as e:
        print(f"Error processing transcript for video {video_id}: {str(e)}")
        return None

def crawl_news_channels(max_videos_per_channel=100):
    """Crawl videos from specified news channels and extract their transcripts"""
    if not verify_api_key():
        return None

    all_data = []
    
    for channel_name, channel_id in NEWS_CHANNELS.items():
        print(f"\nCrawling videos from {channel_name}...")
        
        # Get videos from channel
        videos_df = get_channel_videos(channel_id, max_results=max_videos_per_channel)
        
        if len(videos_df) == 0:
            print(f"No videos found for {channel_name}, skipping...")
            continue
            
        print(f"Found {len(videos_df)} videos, extracting transcripts...")
        
        # Get transcripts for each video
        videos_df['transcript'] = videos_df['video_id'].apply(get_video_transcript)
        
        # Add channel name
        videos_df['channel_name'] = channel_name
        
        # Remove videos without transcripts
        videos_df = videos_df.dropna(subset=['transcript'])
        print(f"Successfully extracted {len(videos_df)} transcripts")
        
        all_data.append(videos_df)
    
    if not all_data:
        print("\nNo data was collected. Please check the errors above.")
        return None
        
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'news_transcripts_{timestamp}.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(final_df)} video transcripts to {output_file}")
    
    return final_df

if __name__ == "__main__":
    print("Starting YouTube News Channel Crawler...")
    crawl_news_channels(max_videos_per_channel=100)
