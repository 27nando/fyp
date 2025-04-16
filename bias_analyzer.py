import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
    # Political and social values
    'freedom', 'patriot', 'traditional', 'faith', 'strong', 'proud',
    'sovereign', 'protection', 'values', 'heritage', 'democracy', 'justice',
    'unity', 'leadership', 'success', 'progress', 'reform', 'prosperity',
    'security', 'peace', 'victory', 'achievement', 'excellence', 'freedom', 
    # Economic terms
    'growth', 'recovery', 'boost', 'flourishing', 'thriving', 'robust',
    'stable', 'profitable', 'sustainable', 'innovative', 'efficient', 'profit',
    # Social impact
    'community', 'family', 'equality', 'diversity', 'inclusive', 'empowerment',
    'opportunity', 'advancement', 'breakthrough', 'transformation', 'innovation'
],
    'negative': [
    # Political criticism
    'radical', 'socialist', 'communist', 'threat', 'dangerous', 'corrupt',
    'invasion', 'chaos', 'crisis', 'disaster', 'extremist', 'authoritarian',
    'tyranny', 'propaganda', 'conspiracy', 'scandal', 'controversy',
    'incompetent', 'failed', 'mismanaged', 'illegitimate',
    # Economic concerns
    'recession', 'inflation', 'crash', 'deficit', 'bankruptcy', 'collapse',
    'burden', 'wasteful', 'devastating', 'costly', 'unstable',
    # Social issues
    'violence', 'crime', 'terrorism', 'discrimination', 'exploitation',
    'oppression', 'inequality', 'poverty', 'conflict', 'crisis'
],
    'emotional': [
    # Negative emotions
    'outrage', 'shocking', 'devastating', 'alarming', 'horrific',
    'terrifying', 'tragic', 'dramatic', 'explosive', 'scandal',
    'catastrophic', 'disastrous', 'nightmare', 'horrendous', 'appalling',
    'disgraceful', 'shameful', 'despicable', 'atrocious',
    # Urgency and intensity
    'breaking', 'exclusive', 'urgent', 'critical', 'crucial',
    'unprecedented', 'massive', 'extreme', 'intense', 'emergency',
    # Fear and anxiety
    'dangerous', 'threatening', 'menacing', 'sinister', 'dire',
    'perilous', 'destructive', 'deadly', 'fatal', 'devastating',
    # Positive intensity
    'extraordinary', 'remarkable', 'spectacular', 'phenomenal',
    'incredible', 'amazing', 'brilliant', 'outstanding', 'magnificent'
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
    
    def analyze_correlations(self):
        """Analyze correlations between different bias metrics"""
        print("Analyzing metric correlations...")
        
        # Prepare correlation features
        correlation_features = [
            'sentiment_score', 'adj_ratio', 'adv_ratio', 'subjective_ratio',
            'bias_word_density', 'emotional_word_density', 'bias_word_variety'
        ]
        
        # Calculate correlation matrix
        self.correlation_matrix = self.df[correlation_features].corr().round(3)
        
        # Calculate specific correlations
        self.feature_correlations = {
            'sentiment_linguistic': {
                'adj_ratio': self.df['sentiment_score'].corr(self.df['adj_ratio']),
                'adv_ratio': self.df['sentiment_score'].corr(self.df['adv_ratio']),
                'subjective_ratio': self.df['sentiment_score'].corr(self.df['subjective_ratio'])
            },
            'bias_patterns': {
                'emotional_density': self.df['bias_word_density'].corr(self.df['emotional_word_density']),
                'sentiment_intensity': abs(self.df['sentiment_score']).corr(self.df['bias_word_density']),
                'subjectivity_impact': self.df['subjective_ratio'].corr(self.df['bias_word_variety'])
            },
            'cluster_correlations': self.analyze_cluster_patterns()
        }
        return self

    def analyze_cluster_patterns(self):
        """Analyze patterns within bias clusters"""
        cluster_stats = {}
        
        for cluster in self.df['bias_cluster'].unique():
            cluster_data = self.df[self.df['bias_cluster'] == cluster]
            cluster_stats[f'cluster_{cluster}'] = {
                'sentiment_std': cluster_data['sentiment_score'].std(),
                'bias_density_mean': cluster_data['bias_word_density'].mean(),
                'subjective_ratio_mean': cluster_data['subjective_ratio'].mean(),
                'size': len(cluster_data)
            }
        
        return cluster_stats

    def plot_bias_comparison(self):
        """Create enhanced visualizations for bias comparison"""
        print("Generating visualizations...")
        
        # Create main analysis plots
        self._plot_main_analysis()
        
        # Create correlation analysis plots
        self._plot_correlation_analysis()
        
        print("Visualizations saved as 'bias_analysis.png' and 'correlation_analysis.png'")

    def _plot_main_analysis(self):
        """Create main bias analysis visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('News Channel Bias Analysis', fontsize=16)
        
        # 1. Enhanced Sentiment Distribution
        sns.violinplot(data=self.df, x='channel_name', y='sentiment_score', ax=axes[0,0])
        axes[0,0].set_title('Sentiment Distribution by Channel')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # 2. Enhanced Loaded Language Usage
        loaded_lang_data = self.channel_metrics[['positive_bias_total', 
                                               'negative_bias_total', 
                                               'emotional_language_total']]
        loaded_lang_data.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Use of Loaded Language by Channel')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1))
        
        # 3. Enhanced POS Pattern Analysis with Error Bars
        pos_data = self.df.groupby('channel_name').agg({
            'adj_ratio': ['mean', 'std'],
            'adv_ratio': ['mean', 'std'],
            'subjective_ratio': ['mean', 'std']
        })
        pos_data['adj_ratio']['mean'].plot(kind='bar', yerr=pos_data['adj_ratio']['std'],
                                          ax=axes[1,0], capsize=5)
        axes[1,0].set_title('Part-of-Speech Patterns by Channel')
        
        # 4. Enhanced Bias Word Frequency with Confidence Intervals
        sns.barplot(data=self.df.melt(id_vars=['channel_name'],
                                     value_vars=['bias_word_density', 'emotional_word_density']),
                    x='channel_name', y='value', hue='variable', ax=axes[1,1])
        axes[1,1].set_title('Bias Word Frequency by Channel')
        
        # 5. Enhanced Cluster Analysis with Proportions
        cluster_props = self.df.groupby(['channel_name', 'bias_cluster']).size().unstack()
        cluster_props = cluster_props.div(cluster_props.sum(axis=1), axis=0)
        cluster_props.plot(kind='bar', stacked=True, ax=axes[2,0])
        axes[2,0].set_title('Proportional Bias Pattern Distribution')
        
        # 6. Enhanced Bias Word Variety with Trend
        sns.boxplot(data=self.df, x='channel_name', y='bias_word_variety', ax=axes[2,1])
        sns.pointplot(data=self.df.groupby('channel_name')['bias_word_variety'].mean().reset_index(),
                      x='channel_name', y='bias_word_variety', color='red', ax=axes[2,1])
        axes[2,1].set_title('Bias Word Variety Distribution')
        
        plt.tight_layout()
        plt.savefig('visualizations/bias_analysis.png', dpi=300, bbox_inches='tight')

    def _plot_correlation_analysis(self):
        """Create correlation analysis visualizations"""
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        sns.heatmap(self.correlation_matrix, annot=True, cmap='RdYlBu', center=0, ax=ax1)
        ax1.set_title('Correlation Matrix of Bias Metrics')
        
        # 2. Sentiment-Subjectivity Relationship
        ax2 = fig.add_subplot(gs[1, 0])
        sns.regplot(data=self.df, x='subjective_ratio', y='sentiment_score', ax=ax2)
        ax2.set_title('Sentiment vs Subjectivity')
        
        # 3. Bias-Emotion Relationship
        ax3 = fig.add_subplot(gs[1, 1])
        sns.regplot(data=self.df, x='bias_word_density', y='emotional_word_density', ax=ax3)
        ax3.set_title('Bias vs Emotional Content')
        
        # 4. Cluster Characteristics
        ax4 = fig.add_subplot(gs[2, :])
        cluster_data = pd.DataFrame(self.feature_correlations['cluster_correlations']).T
        cluster_data[['sentiment_std', 'bias_density_mean', 'subjective_ratio_mean']].plot(kind='bar', ax=ax4)
        ax4.set_title('Cluster Characteristics')
        
        plt.tight_layout()
        plt.savefig('visualizations/correlation_analysis.png', dpi=300, bbox_inches='tight')
        
        # Save individual visualizations
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='channel_name', y='sentiment_score')
        plt.title('Sentiment Distribution by Channel')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(10, 6))
        loaded_lang_data = self.channel_metrics[['positive_bias_total', 'negative_bias_total', 'emotional_language_total']]
        loaded_lang_data.plot(kind='bar')
        plt.title('Loaded Language Usage by Channel')
        plt.tight_layout()
        plt.savefig('visualizations/loaded_language_usage.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(10, 6))
        pos_data = self.df.groupby('channel_name')[['adj_ratio', 'adv_ratio', 'subjective_ratio']].mean()
        pos_data.plot(kind='bar')
        plt.title('POS Patterns by Channel')
        plt.tight_layout()
        plt.savefig('visualizations/pos_patterns.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(10, 6))
        freq_data = self.df.groupby('channel_name')[['bias_word_density', 'emotional_word_density']].mean()
        freq_data.plot(kind='bar')
        plt.title('Bias Word Frequency by Channel')
        plt.tight_layout()
        plt.savefig('visualizations/bias_frequency.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(10, 6))
        cluster_props = self.df.groupby(['channel_name', 'bias_cluster']).size().unstack()
        cluster_props = cluster_props.div(cluster_props.sum(axis=1), axis=0)
        cluster_props.plot(kind='bar', stacked=True)
        plt.title('Bias Pattern Clusters by Channel')
        plt.tight_layout()
        plt.savefig('visualizations/bias_clusters.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='channel_name', y='bias_word_variety')
        plt.title('Bias Word Variety by Channel')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/bias_variety.png', dpi=300, bbox_inches='tight')
        
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
        
        print("\n5. Correlation Analysis:")
        print("-" * 30)
        
        # Print sentiment-linguistic correlations
        print("\nSentiment-Linguistic Correlations:")
        for metric, corr in self.feature_correlations['sentiment_linguistic'].items():
            print(f"{metric}: {corr:.3f}")
        
        # Print bias pattern correlations
        print("\nBias Pattern Correlations:")
        for metric, corr in self.feature_correlations['bias_patterns'].items():
            print(f"{metric}: {corr:.3f}")
        
        # Print cluster insights
        print("\nCluster Characteristics:")
        for cluster, stats in self.feature_correlations['cluster_correlations'].items():
            print(f"\n{cluster}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.3f}")
        
        print("\n6. Key Findings:")
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
        
        # Strong correlations
        strong_correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr >= 0.6:  # Strong correlation threshold
                    strong_correlations.append({
                        'features': (self.correlation_matrix.columns[i],
                                    self.correlation_matrix.columns[j]),
                        'correlation': corr
                    })
        
        if strong_correlations:
            print("\nStrong Feature Correlations:")
            for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"{corr['features'][0]} - {corr['features'][1]}: {corr['correlation']:.3f}")
        
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

    def save_report_to_pdf(self, report_type='channel_analysis'):
        """Save the analysis report to a PDF file"""
    
        if report_type == 'channel_analysis':
            filename = 'reports/channel_analysis.pdf'
            title = 'Channel-wise Bias Analysis Report'
            content = [
                ('Channel Metrics', self.channel_metrics.round(3).to_string()),
                ('POS Pattern Analysis', self.df.groupby('channel_name')[['adj_ratio', 'adv_ratio', 'subjective_ratio']].mean().round(3).to_string()),
                ('Bias Pattern Clusters', self.cluster_stats.round(3).to_string())
            ]
        elif report_type == 'comparative_report':
            filename = 'reports/comparative_report.pdf'
            title = 'Cross-Channel Comparative Analysis'
            content = [
                ('Sentiment Comparison', self.channel_metrics[['sentiment_mean', 'sentiment_std']].round(3).to_string()),
                ('Loaded Language Comparison', self.channel_metrics[['positive_bias_total', 'negative_bias_total', 'emotional_language_total']].round(3).to_string()),
                ('Bias Ratio Comparison', self.channel_metrics[['bias_ratio']].round(3).to_string())
            ]
        elif report_type == 'trend_analysis':
            filename = 'reports/trend_analysis.pdf'
            title = 'Temporal Bias Pattern Analysis'
            content = [
                ('Correlation Analysis', pd.DataFrame(self.feature_correlations['sentiment_linguistic'], index=['correlation']).round(3).to_string()),
                ('Bias Patterns', pd.DataFrame(self.feature_correlations['bias_patterns'], index=['correlation']).round(3).to_string()),
                ('Cluster Analysis', pd.DataFrame(self.feature_correlations['cluster_correlations']).T.round(3).to_string())
            ]
        else:  # executive_summary
            filename = 'reports/executive_summary.pdf'
            title = 'Media Bias Analysis: Executive Summary'
            content = [
                ('Key Findings', f"Most emotional: {self.channel_metrics['emotional_language_total'].idxmax()}\n" +
                               f"Most positive: {self.channel_metrics['sentiment_mean'].idxmax()}\n" +
                               f"Most negative: {self.channel_metrics['sentiment_mean'].idxmin()}\n" +
                               f"Highest bias ratio: {self.channel_metrics['bias_ratio'].idxmax()}"),
                ('Strong Correlations', self.get_strong_correlations())
            ]
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Add content
        for section_title, section_content in content:
            story.append(Paragraph(section_title, styles['Heading2']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(section_content.replace('\n', '<br/>'), styles['BodyText']))
            story.append(Spacer(1, 20))
        
        doc.build(story)
        print(f"Report saved as {filename}")

    def get_strong_correlations(self):
        """Get a formatted string of strong correlations"""
        strong_correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr >= 0.6:
                    strong_correlations.append(
                        f"{self.correlation_matrix.columns[i]} - {self.correlation_matrix.columns[j]}: {corr:.3f}"
                    )
        return '\n'.join(strong_correlations)

def main():
    # Find the latest combined dataset from speech crawler
    import os
    import glob
    
    # Create necessary directories
    for directory in ['data', 'reports', 'visualizations']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created {directory} directory")
    
    # Get the latest file that matches the pattern
    data_files = glob.glob('data/all_channels_*.csv')
    if not data_files:
        print("Error: No data files found in data/ directory")
        return
    
    data_file = max(data_files)  # Gets the latest file by timestamp
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
    
    # Run correlation analysis
    analyzer.analyze_correlations()
    
    # Generate visualizations and reports
    analyzer.plot_bias_comparison()
    analyzer.generate_report()
    
    # Generate PDF reports
    analyzer.save_report_to_pdf('channel_analysis')
    analyzer.save_report_to_pdf('comparative_report')
    analyzer.save_report_to_pdf('trend_analysis')
    analyzer.save_report_to_pdf('executive_summary')

if __name__ == "__main__":
    main()
