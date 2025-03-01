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

# List of news channels to analyze
NEWS_CHANNELS = {
    'Fox News': 'UCXIJgqnII2ZOINSWNOGFThA',
    'CNN': 'UCupvZG-5ko_eiXAupbDfxWw',
    'MSNBC': 'UCaXkIU1QidjPwiAYu6GcHjg',
    'NewsMax': 'UCx6h-dWzJ5NpAlja1YsApdg',
    'OAN': 'UCe02lGcO-ahAURWuxAJnjdA'
}

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

def get_channel_videos(channel_id, max_results=100):
    """Get videos from a specific channel"""
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        
        search_response = youtube.search().list(
            channelId=channel_id,
            part='id,snippet',
            order='date',  # Get most recent videos
            type='video',
            maxResults=min(max_results, 50)
        ).execute()

        videos = []
        while len(videos) < max_results:
            for item in search_response.get('items', []):
                video_data = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'description': item['snippet']['description']
                }
                videos.append(video_data)
                
            if 'nextPageToken' in search_response and len(videos) < max_results:
                search_response = youtube.search().list(
                    channelId=channel_id,
                    part='id,snippet',
                    order='date',
                    type='video',
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
    """Extract transcript for a given video ID"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        return full_transcript
    except Exception as e:
        print(f"Could not get transcript for video {video_id}: {str(e)}")
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
