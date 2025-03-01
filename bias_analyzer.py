import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for all plots
plt.style.use('default')
sns.set_theme()

# Load spaCy model for text processing
nlp = spacy.load('en_core_web_sm')

# Load pre-defined bias-indicating words
LOADED_WORDS = {
    'positive': [
        'freedom', 'patriot', 'traditional', 'faith', 'strong', 'proud',
        'sovereign', 'protection', 'values', 'heritage'
    ],
    'negative': [
        'radical', 'socialist', 'communist', 'threat', 'dangerous', 'corrupt',
        'invasion', 'chaos', 'crisis', 'disaster'
    ],
    'emotional': [
        'outrage', 'shocking', 'devastating', 'alarming', 'horrific',
        'terrifying', 'tragic', 'dramatic', 'explosive', 'scandal'
    ]
}

class BiasAnalyzer:
    def __init__(self, csv_file):
        """Initialize the BiasAnalyzer with a CSV file containing news transcripts"""
        self.df = pd.read_csv(csv_file)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()
    
    def analyze_sentiment(self):
        """Analyze sentiment of transcripts"""
        print("Analyzing sentiment...")
        self.df['sentiment_score'] = self.df['transcript'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        return self
    
    def detect_loaded_language(self):
        """Detect loaded language and emotional words"""
        print("Detecting loaded language...")
        
        def count_loaded_words(text):
            text = str(text).lower()
            counts = {
                'positive_bias': sum(1 for word in LOADED_WORDS['positive'] if word in text),
                'negative_bias': sum(1 for word in LOADED_WORDS['negative'] if word in text),
                'emotional_language': sum(1 for word in LOADED_WORDS['emotional'] if word in text)
            }
            return pd.Series(counts)
        
        loaded_word_counts = self.df['transcript'].apply(count_loaded_words)
        self.df = pd.concat([self.df, loaded_word_counts], axis=1)
        return self
    
    def analyze_topics(self, n_topics=5):
        """Perform topic modeling on transcripts"""
        print("Analyzing topics...")
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf = vectorizer.fit_transform(self.df['transcript'].fillna(''))
        
        # Perform LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        self.topics = topics
        return self
    
    def calculate_bias_metrics(self):
        """Calculate overall bias metrics for each channel"""
        print("Calculating bias metrics...")
        
        channel_metrics = self.df.groupby('channel_name').agg({
            'sentiment_score': ['mean', 'std'],
            'positive_bias': 'sum',
            'negative_bias': 'sum',
            'emotional_language': 'sum'
        }).round(3)
        
        # Flatten column names
        channel_metrics.columns = ['sentiment_mean', 'sentiment_std', 
                                 'positive_bias_total', 'negative_bias_total',
                                 'emotional_language_total']
        
        # Calculate bias ratio (positive vs negative bias)
        channel_metrics['bias_ratio'] = (
            channel_metrics['positive_bias_total'] / 
            channel_metrics['negative_bias_total']
        ).round(3)
        
        self.channel_metrics = channel_metrics
        return self
    
    def plot_bias_comparison(self):
        """Create visualizations for bias comparison"""
        print("Generating visualizations...")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('News Channel Bias Analysis', fontsize=16)
        
        # 1. Sentiment Distribution
        sns.boxplot(data=self.df, x='channel_name', y='sentiment_score', ax=axes[0,0])
        axes[0,0].set_title('Sentiment Distribution by Channel')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # 2. Loaded Language Usage
        loaded_lang_data = self.channel_metrics[['positive_bias_total', 
                                               'negative_bias_total', 
                                               'emotional_language_total']]
        loaded_lang_data.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Use of Loaded Language by Channel')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        
        # 3. Bias Ratio
        self.channel_metrics['bias_ratio'].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Bias Ratio (Positive/Negative) by Channel')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # 4. Emotional Language Usage
        emotional_data = self.channel_metrics['emotional_language_total']
        emotional_data.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Emotional Language Usage by Channel')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('bias_analysis.png')
        print("Visualizations saved as 'bias_analysis.png'")
        
    def generate_report(self):
        """Generate a comprehensive bias analysis report"""
        print("\nBias Analysis Report")
        print("=" * 50)
        
        print("\n1. Channel Bias Metrics:")
        print("-" * 30)
        print(self.channel_metrics)
        
        print("\n2. Identified Topics:")
        print("-" * 30)
        for topic in self.topics:
            print(topic)
        
        print("\n3. Key Findings:")
        print("-" * 30)
        
        # Most biased channel (based on emotional language)
        most_emotional = self.channel_metrics['emotional_language_total'].idxmax()
        print(f"Most emotionally charged language: {most_emotional}")
        
        # Most positive and negative channels
        most_positive = self.channel_metrics['sentiment_mean'].idxmax()
        most_negative = self.channel_metrics['sentiment_mean'].idxmin()
        print(f"Most positive sentiment: {most_positive}")
        print(f"Most negative sentiment: {most_negative}")
        
        # Channel with highest bias ratio
        most_biased = self.channel_metrics['bias_ratio'].idxmax()
        print(f"Highest positive-to-negative bias ratio: {most_biased}")

def main():
    # Initialize analyzer with your CSV file
    analyzer = BiasAnalyzer('news_transcripts_20250301_163102.csv')
    
    # Run analysis pipeline
    analyzer.analyze_sentiment()
    analyzer.detect_loaded_language()
    analyzer.analyze_topics()
    analyzer.calculate_bias_metrics()
    
    # Generate visualizations and report
    analyzer.plot_bias_comparison()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
