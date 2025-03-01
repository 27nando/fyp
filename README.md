# News Media Bias Analysis

This project analyzes bias in news media by crawling YouTube news channels, extracting video transcripts, and performing natural language processing (NLP) to detect various forms of bias in the reporting.

## Features

- YouTube news channel crawler
- Automatic transcript extraction
- Comprehensive bias analysis including:
  - Sentiment analysis
  - Loaded language detection
  - Topic modeling
  - Bias metrics visualization

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Set up YouTube API:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable the YouTube Data API v3
   - Create API credentials
   - Create a `.env` file in the project root and add your API key:
     ```
     YOUTUBE_API_KEY=your_api_key_here
     ```

## Usage

1. Crawl YouTube news channels:
```bash
python youtube_crawler.py
```
This will:
- Fetch videos from specified news channels
- Extract available transcripts
- Save the data to a CSV file

2. Analyze bias in the transcripts:
```bash
python bias_analyzer.py
```
This will:
- Perform sentiment analysis
- Detect loaded language
- Generate topic models
- Create visualizations
- Output a comprehensive bias report

## Analysis Components

### Sentiment Analysis
- Measures the overall sentiment (positive/negative) of news coverage
- Uses TextBlob for sentiment scoring

### Loaded Language Detection
Tracks the usage of:
- Positive bias words
- Negative bias words
- Emotional language

### Topic Modeling
- Uses Latent Dirichlet Allocation (LDA)
- Identifies key themes in news coverage
- Helps track focus areas of different channels

### Visualization
Generates plots showing:
- Sentiment distribution by channel
- Use of loaded language
- Bias ratios
- Emotional language usage

## Output

The analysis produces:
1. A CSV file containing raw transcript data
2. Visual plots saved as 'bias_analysis.png'
3. A detailed report showing:
   - Channel-wise bias metrics
   - Identified topics
   - Key findings about bias patterns

## Dependencies

- youtube-dl
- youtube-transcript-api
- pandas
- python-dotenv
- textblob
- transformers
- scikit-learn
- spacy
- matplotlib
- seaborn
- torch

## Notes

- Some videos may not have available transcripts
- Analysis results should be interpreted in context
- Bias detection is based on predefined metrics and may not capture all forms of media bias

## Contributing

Feel free to:
- Report issues
- Submit pull requests
- Suggest improvements to bias detection algorithms
- Add new analysis features
