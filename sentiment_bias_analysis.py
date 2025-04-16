import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from bias_analyzer import BiasAnalyzer

class SentimentBiasAnalyzer:
    def __init__(self, csv_file):
        """Initialize with the news transcript data"""
        self.analyzer = BiasAnalyzer(csv_file)
        # Run initial analysis
        self.analyzer.analyze_sentiment().detect_loaded_language().analyze_pos_patterns().analyze_frequency_patterns()
        self.df = self.analyzer.df
        
    def analyze_sentiment_bias_relationship(self):
        """Analyze the relationship between sentiment and various bias metrics"""
        print("Analyzing sentiment-bias relationships...")
        
        # 1. Calculate absolute sentiment (intensity regardless of direction)
        self.df['sentiment_intensity'] = abs(self.df['sentiment_score'])
        
        # 2. Calculate bias intensity
        self.df['bias_intensity'] = (self.df['positive_bias'] + self.df['negative_bias']) / \
                                   self.df['total_words'].clip(lower=1)
        
        # 3. Calculate correlations
        correlations = {
            'bias_sentiment': {
                'sentiment_vs_bias': stats.pearsonr(
                    self.df['sentiment_score'], 
                    self.df['bias_intensity']
                ),
                'sentiment_vs_emotional': stats.pearsonr(
                    self.df['sentiment_score'], 
                    self.df['emotional_language']
                ),
                'sentiment_vs_subjectivity': stats.pearsonr(
                    self.df['sentiment_score'], 
                    self.df['subjective_ratio']
                )
            }
        }
        
        # 4. Channel-wise analysis
        channel_analysis = self.df.groupby('channel_name').agg({
            'sentiment_score': ['mean', 'std'],
            'bias_intensity': ['mean', 'std'],
            'emotional_language': ['mean', 'std'],
            'subjective_ratio': ['mean', 'std']
        }).round(3)
        
        # 5. Sentiment polarity vs bias type analysis
        self.df['sentiment_category'] = pd.cut(
            self.df['sentiment_score'],
            bins=[-1, -0.3, 0.3, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        bias_by_sentiment = self.df.groupby('sentiment_category').agg({
            'positive_bias': 'mean',
            'negative_bias': 'mean',
            'emotional_language': 'mean',
            'bias_word_variety': 'mean'
        }).round(3)
        
        return correlations, channel_analysis, bias_by_sentiment
    
    def visualize_relationships(self):
        """Create visualizations for sentiment-bias relationships"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sentiment vs Bias Intensity
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=self.df,
            x='sentiment_score',
            y='bias_intensity',
            hue='channel_name',
            alpha=0.6,
            ax=ax1
        )
        ax1.set_title('Sentiment Score vs Bias Intensity')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Bias Intensity')
        
        # 2. Emotional Language by Sentiment Category
        ax2 = plt.subplot(2, 2, 2)
        sentiment_emotional = self.df.groupby('sentiment_category')['emotional_language'].mean()
        sentiment_emotional.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Emotional Language Usage by Sentiment')
        ax2.set_xlabel('Sentiment Category')
        ax2.set_ylabel('Average Emotional Words')
        
        # 3. Channel-wise Sentiment vs Bias
        ax3 = plt.subplot(2, 2, 3)
        channel_metrics = self.df.groupby('channel_name').agg({
            'sentiment_score': 'mean',
            'bias_intensity': 'mean'
        })
        sns.scatterplot(
            data=channel_metrics,
            x='sentiment_score',
            y='bias_intensity',
            s=100,
            ax=ax3
        )
        for idx, row in channel_metrics.iterrows():
            ax3.annotate(idx, (row['sentiment_score'], row['bias_intensity']))
        ax3.set_title('Channel-wise Sentiment vs Bias')
        
        # 4. Bias Type Distribution by Sentiment
        ax4 = plt.subplot(2, 2, 4)
        bias_by_sent = self.df.groupby('sentiment_category')[
            ['positive_bias', 'negative_bias', 'emotional_language']
        ].mean()
        bias_by_sent.plot(kind='bar', ax=ax4)
        ax4.set_title('Bias Type Distribution by Sentiment Category')
        ax4.legend(bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_bias_relationship.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'sentiment_bias_relationship.png'")

def main():
    analyzer = SentimentBiasAnalyzer('processed_transcripts.csv')
    correlations, channel_analysis, bias_by_sentiment = analyzer.analyze_sentiment_bias_relationship()
    
    print("\nCorrelations between Sentiment and Bias Metrics:")
    for metric, (corr, p_val) in correlations['bias_sentiment'].items():
        print(f"{metric}: correlation = {corr:.3f}, p-value = {p_val:.3f}")
    
    print("\nChannel-wise Analysis:")
    print(channel_analysis)
    
    print("\nBias Metrics by Sentiment Category:")
    print(bias_by_sentiment)
    
    analyzer.visualize_relationships()

if __name__ == "__main__":
    main()
