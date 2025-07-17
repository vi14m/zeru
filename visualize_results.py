import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_credit_scores():
    """Load credit scores from CSV file"""
    file_path = 'wallet_credit_scores.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run credit_score_model.py first.")
        return None
    
    return pd.read_csv(file_path, index_col=0)

def visualize_score_distribution(scores_df):
    """Visualize the distribution of credit scores"""
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    sns.histplot(scores_df['credit_score'], kde=True, bins=20)
    
    # Add vertical lines for score ranges
    plt.axvline(x=250, color='r', linestyle='--', alpha=0.7, label='High Risk (0-250)')
    plt.axvline(x=500, color='y', linestyle='--', alpha=0.7, label='Medium Risk (250-500)')
    plt.axvline(x=750, color='g', linestyle='--', alpha=0.7, label='Low Risk (500-750)')
    
    # Add labels and title
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.title('Distribution of Wallet Credit Scores')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    plt.close()

def create_risk_categories(scores_df):
    """Create risk categories based on credit scores"""
    # Define risk categories
    conditions = [
        (scores_df['credit_score'] < 250),
        (scores_df['credit_score'] >= 250) & (scores_df['credit_score'] < 500),
        (scores_df['credit_score'] >= 500) & (scores_df['credit_score'] < 750),
        (scores_df['credit_score'] >= 750)
    ]
    categories = np.array(['High Risk', 'Medium-High Risk', 'Medium-Low Risk', 'Low Risk'])
    
    # Create new column with risk categories
    scores_df['risk_category'] = np.select(conditions, categories, default='Unknown')
    
    return scores_df

def visualize_risk_categories(scores_df):
    """Visualize the distribution of risk categories"""
    # Count wallets in each risk category
    risk_counts = scores_df['risk_category'].value_counts().sort_index()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    colors = ['#FF5733', '#FFC300', '#33FF57', '#3380FF']
    
    # Create explode list with the same length as risk_counts
    explode = [0] * len(risk_counts)
    # Set explode values for high and medium-high risk categories if they exist
    categories = list(risk_counts.index)
    if 'High Risk' in categories:
        explode[categories.index('High Risk')] = 0.1
    if 'Medium-High Risk' in categories:
        explode[categories.index('Medium-High Risk')] = 0.05
    
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=colors[:len(risk_counts)], explode=explode)
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Wallet Risk Categories Distribution')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('risk_categories.png')
    plt.close()

def main():
    # Load credit scores
    scores_df = load_credit_scores()
    if scores_df is None:
        return
    
    print(f"Loaded credit scores for {len(scores_df)} wallets")
    
    # Visualize score distribution
    print("Generating score distribution visualization...")
    visualize_score_distribution(scores_df)
    
    # Create and visualize risk categories
    print("Generating risk categories visualization...")
    scores_df = create_risk_categories(scores_df)
    visualize_risk_categories(scores_df)
    
    # # Save categorized scores
    # scores_df.to_csv('wallet_risk_categories.csv')
    print("Visualizations and categorized scores saved.")
    
    # Display statistics
    print("\nRisk Category Statistics:")
    print(scores_df['risk_category'].value_counts().sort_index())

if __name__ == "__main__":
    main()