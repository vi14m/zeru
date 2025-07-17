import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_credit_scores(file_path):
    """Load credit scores from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    return pd.read_csv(file_path, index_col=0)

def analyze_score_ranges(scores_df, output_file):
    """Analyze credit scores by ranges and save to markdown file"""
    # Define score ranges
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
              (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
    
    # Count wallets in each range
    range_counts = []
    for low, high in ranges:
        count = ((scores_df['credit_score'] >= low) & (scores_df['credit_score'] < high)).sum()
        range_counts.append({
            'range': f"{low}-{high}",
            'count': count,
            'percentage': count / len(scores_df) * 100
        })
    
    # Create DataFrame for ranges
    range_df = pd.DataFrame(range_counts)
    
    # Visualize score ranges
    plt.figure(figsize=(12, 6))
    sns.barplot(x='range', y='count', data=range_df)
    plt.title('Distribution of Wallet Credit Scores by Range')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('score_ranges.png')
    plt.close()
    
    # Define risk categories
    risk_categories = {
        'High Risk': (0, 250),
        'Medium-High Risk': (250, 500),
        'Medium-Low Risk': (500, 750),
        'Low Risk': (750, 1000)
    }
    
    # Analyze wallets by risk category
    with open(output_file, 'w') as f:
        f.write("# DeFi Wallet Credit Score Analysis\n\n")
        
        f.write("## Score Distribution\n\n")
        f.write("![Score Distribution](score_distribution.png)\n\n")
        
        f.write("## Score Distribution by Range\n\n")
        f.write("![Score Ranges](score_ranges.png)\n\n")
        
        f.write("### Score Range Analysis\n\n")
        f.write("| Score Range | Number of Wallets | Percentage |\n")
        f.write("|-------------|------------------|------------|\n")
        for row in range_df.itertuples():
            f.write(f"| {row.range} | {row.count} | {row.percentage:.2f}% |\n")
        f.write("\n")
        
        f.write("## Risk Category Distribution\n\n")
        f.write("![Risk Categories](risk_categories.png)\n\n")
        
        # Analyze each risk category
        for category, (low, high) in risk_categories.items():
            category_wallets = scores_df[(scores_df['credit_score'] >= low) & (scores_df['credit_score'] < high)]
            f.write(f"### {category} Wallets ({low}-{high})\n\n")
            
            if len(category_wallets) == 0:
                f.write(f"No wallets found in this category.\n\n")
                continue
            
            f.write(f"**Number of Wallets:** {len(category_wallets)} ({len(category_wallets)/len(scores_df)*100:.2f}%)\n\n")
            
            # Calculate statistics for this category
            f.write("**Statistics:**\n\n")
            f.write(f"- Average Score: {category_wallets['credit_score'].mean():.2f}\n")
            f.write(f"- Median Score: {category_wallets['credit_score'].median():.2f}\n")
            f.write(f"- Min Score: {category_wallets['credit_score'].min()}\n")
            f.write(f"- Max Score: {category_wallets['credit_score'].max()}\n\n")
            
            # Analyze behavior patterns if we have additional features
            if 'risk_category' in category_wallets.columns:
                f.write("**Behavior Patterns:**\n\n")
                
                if category == 'High Risk' or category == 'Medium-High Risk':
                    f.write("Wallets in this category typically exhibit:\n\n")
                    f.write("- Low repayment rates relative to borrowing\n")
                    f.write("- Insufficient collateralization\n")
                    f.write("- History of liquidations\n")
                    f.write("- Short account age or sporadic activity\n")
                    f.write("- Limited asset diversity\n\n")
                else:
                    f.write("Wallets in this category typically exhibit:\n\n")
                    f.write("- High repayment rates relative to borrowing\n")
                    f.write("- Strong collateralization ratios\n")
                    f.write("- No history of liquidations\n")
                    f.write("- Longer account age with consistent activity\n")
                    f.write("- Greater asset diversity\n\n")

def analyze_ml_scores(scores_df, output_file):
    """Analyze ML-based credit scores and append to markdown file"""
    # Visualize ML score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(scores_df['credit_score'], kde=True, bins=20)
    plt.axvline(x=250, color='r', linestyle='--', alpha=0.7, label='High Risk (0-250)')
    plt.axvline(x=500, color='y', linestyle='--', alpha=0.7, label='Medium Risk (250-500)')
    plt.axvline(x=750, color='g', linestyle='--', alpha=0.7, label='Low Risk (500-750)')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.title('Distribution of Wallet Credit Scores (Machine Learning Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ml_score_ranges.png')
    plt.close()
    
    # Append ML analysis to the markdown file
    with open(output_file, 'a') as f:
        f.write("## Machine Learning Model Analysis\n\n")
        f.write("![ML Score Distribution](ml_score_distribution.png)\n\n")
        f.write("![ML Score Ranges](ml_score_ranges.png)\n\n")
        
        f.write("### Feature Importance\n\n")
        f.write("![Feature Importance](feature_importance.png)\n\n")
        
        f.write("### Model Comparison\n\n")
        f.write("The machine learning model provides a more nuanced credit scoring approach compared to the rule-based model:\n\n")
        f.write("1. **Feature Weighting:** The ML model automatically determines optimal feature weights based on patterns in the data\n")
        f.write("2. **Complex Relationships:** Captures non-linear relationships between features\n")
        f.write("3. **Adaptability:** Can adapt to changing patterns as new data becomes available\n\n")

def main():
    # Analyze rule-based model scores
    print("Analyzing rule-based model scores...")
    rule_scores = load_credit_scores('wallet_credit_scores.csv')
    if rule_scores is not None:
        analyze_score_ranges(rule_scores, 'analysis.md')
    
    # Analyze ML model scores if available
    print("Analyzing machine learning model scores...")
    ml_scores = load_credit_scores('wallet_credit_scores_ml.csv')
    if ml_scores is not None:
        analyze_ml_scores(ml_scores, 'analysis.md')
    
    print("Analysis complete. Results saved to analysis.md")

if __name__ == "__main__":
    main()