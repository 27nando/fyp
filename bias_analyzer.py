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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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
        self.scaler = StandardScaler()
        
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
    
    def analyze_pos_patterns(self):
        """Analyze part-of-speech patterns for bias detection"""
        print("Analyzing POS patterns...")
        
        def extract_pos_features(text):
            doc = nlp(str(text))
            # Count POS tags
            pos_counts = Counter([token.pos_ for token in doc])
            # Calculate ratios of different POS tags
            total_tokens = len(doc)
            if total_tokens == 0:
                return pd.Series({
                    'adj_ratio': 0,
                    'adv_ratio': 0,
                    'verb_ratio': 0,
                    'noun_ratio': 0,
                    'subjective_ratio': 0
                })
            
            # Calculate POS ratios
            adj_ratio = pos_counts.get('ADJ', 0) / total_tokens
            adv_ratio = pos_counts.get('ADV', 0) / total_tokens
            verb_ratio = pos_counts.get('VERB', 0) / total_tokens
            noun_ratio = pos_counts.get('NOUN', 0) / total_tokens
            
            # Subjective language ratio (adjectives + adverbs)
            subjective_ratio = (pos_counts.get('ADJ', 0) + pos_counts.get('ADV', 0)) / total_tokens
            
            return pd.Series({
                'adj_ratio': adj_ratio,
                'adv_ratio': adv_ratio,
                'verb_ratio': verb_ratio,
                'noun_ratio': noun_ratio,
                'subjective_ratio': subjective_ratio
            })
        
        pos_features = self.df['transcript'].apply(extract_pos_features)
        self.df = pd.concat([self.df, pos_features], axis=1)
        return self
    
    def analyze_bias_patterns(self, n_components=3):
        """Analyze bias patterns using Gaussian Mixture Models"""
        print("Analyzing bias patterns...")
        
        # Prepare features for GMM
        features = self.df[['sentiment_score', 'positive_bias', 'negative_bias', 
                          'emotional_language', 'adj_ratio', 'adv_ratio', 
                          'subjective_ratio']].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.df['bias_cluster'] = gmm.fit_predict(scaled_features)
        
        # Calculate cluster characteristics
        cluster_stats = []
        for i in range(n_components):
            cluster_data = self.df[self.df['bias_cluster'] == i]
            stats = {
                'cluster': i,
                'size': len(cluster_data),
                'avg_sentiment': cluster_data['sentiment_score'].mean(),
                'avg_subjectivity': cluster_data['subjective_ratio'].mean(),
                'emotional_content': cluster_data['emotional_language'].mean()
            }
            cluster_stats.append(stats)
        
        self.cluster_stats = pd.DataFrame(cluster_stats)
        return self
    
    def analyze_frequency_patterns(self):
        """Analyze frequency patterns of biased language"""
        print("Analyzing frequency patterns...")
        
        def calculate_frequency_metrics(text):
            words = str(text).lower().split()
            if not words:
                return pd.Series({
                    'bias_word_density': 0,
                    'emotional_word_density': 0,
                    'bias_word_variety': 0
                })
            
            # Count unique and total bias words
            bias_words = set(LOADED_WORDS['positive'] + LOADED_WORDS['negative'])
            emotional_words = set(LOADED_WORDS['emotional'])
            
            bias_word_count = sum(1 for word in words if word in bias_words)
            emotional_word_count = sum(1 for word in words if word in emotional_words)
            
            unique_bias_words = len(set(word for word in words if word in bias_words))
            
            # Calculate metrics
            bias_word_density = bias_word_count / len(words)
            emotional_word_density = emotional_word_count / len(words)
            bias_word_variety = unique_bias_words / max(1, bias_word_count)
            
            return pd.Series({
                'bias_word_density': bias_word_density,
                'emotional_word_density': emotional_word_density,
                'bias_word_variety': bias_word_variety
            })
        
        frequency_metrics = self.df['transcript'].apply(calculate_frequency_metrics)
        self.df = pd.concat([self.df, frequency_metrics], axis=1)
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
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
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
        
        # 3. POS Pattern Analysis
        pos_data = self.df.groupby('channel_name')[['adj_ratio', 'adv_ratio', 'subjective_ratio']].mean()
        pos_data.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Part-of-Speech Patterns by Channel')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # 4. Bias Word Frequency Analysis
        freq_data = self.df.groupby('channel_name')[['bias_word_density', 'emotional_word_density']].mean()
        freq_data.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Bias Word Frequency by Channel')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        # 5. Bias Clusters
        cluster_sizes = self.df.groupby(['channel_name', 'bias_cluster']).size().unstack()
        cluster_sizes.plot(kind='bar', stacked=True, ax=axes[2,0])
        axes[2,0].set_title('Bias Pattern Clusters by Channel')
        axes[2,0].set_xticklabels(axes[2,0].get_xticklabels(), rotation=45)
        
        # 6. Bias Word Variety
        sns.boxplot(data=self.df, x='channel_name', y='bias_word_variety', ax=axes[2,1])
        axes[2,1].set_title('Bias Word Variety by Channel')
        axes[2,1].set_xticklabels(axes[2,1].get_xticklabels(), rotation=45)
        
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
        
        print("\n3. POS Pattern Analysis:")
        print("-" * 30)
        pos_summary = self.df.groupby('channel_name')[['adj_ratio', 'adv_ratio', 'subjective_ratio']].mean()
        print(pos_summary.round(3))
        
        print("\n4. Bias Pattern Clusters:")
        print("-" * 30)
        print(self.cluster_stats.round(3))
        
        print("\n5. Key Findings:")
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
        
        # Channel with highest subjective language
        most_subjective = self.df.groupby('channel_name')['subjective_ratio'].mean().idxmax()
        print(f"Most subjective language use: {most_subjective}")
        
        # Most common bias pattern
        dominant_cluster = self.df['bias_cluster'].mode().iloc[0]
        cluster_char = self.cluster_stats.loc[self.cluster_stats['cluster'] == dominant_cluster].iloc[0]
        print(f"\nDominant bias pattern (Cluster {dominant_cluster}):")
        print(f"- Size: {cluster_char['size']} segments")
        print(f"- Average sentiment: {cluster_char['avg_sentiment']:.3f}")
        print(f"- Average subjectivity: {cluster_char['avg_subjectivity']:.3f}")
        print(f"- Emotional content: {cluster_char['emotional_content']:.3f}")

def main():
    # Use the latest combined dataset from speech crawler
    data_file = 'data/all_channels_20250311_151308.csv'
    
    print(f"Running bias analysis on {data_file}...")
    analyzer = BiasAnalyzer(data_file)
    
    # Run analysis pipeline
    analyzer.analyze_sentiment()
    analyzer.detect_loaded_language()
    analyzer.analyze_topics()
    analyzer.analyze_pos_patterns()
    analyzer.analyze_frequency_patterns()
    analyzer.analyze_bias_patterns()
    analyzer.calculate_bias_metrics()
    
    # Generate visualizations and report
    analyzer.plot_bias_comparison()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
