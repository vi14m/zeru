import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os
from visualize_results import create_risk_categories

# Define the path to the JSON file
JSON_FILE_PATH = 'user-wallet-transactions.json'

def load_transactions():
    """Load transactions from JSON file"""
    try:
        with open(JSON_FILE_PATH, 'r') as file:
            transactions = json.load(file)
        print(f"Loaded {len(transactions)} transactions")
        return transactions
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return []

def extract_features(transactions):
    """Extract features from transactions"""
    # Group transactions by wallet
    wallets = {}
    
    for tx in transactions:
        wallet = tx.get('userWallet')
        if not wallet:
            continue
            
        if wallet not in wallets:
            wallets[wallet] = {
                'deposit_count': 0,
                'deposit_amount_total': 0,
                'deposit_amount_avg': 0,
                'borrow_count': 0,
                'borrow_amount_total': 0,
                'borrow_amount_avg': 0,
                'repay_count': 0,
                'repay_amount_total': 0,
                'repay_amount_avg': 0,
                'redeem_count': 0,
                'redeem_amount_total': 0,
                'redeem_amount_avg': 0,
                'liquidation_count': 0,
                'liquidation_amount_total': 0,
                'liquidation_amount_avg': 0,
                'unique_assets': set(),
                'first_tx_timestamp': float('inf'),
                'last_tx_timestamp': 0,
                'tx_count': 0,
                'is_liquidated': False,
                'repay_to_borrow_ratio': 0,
                'deposit_to_borrow_ratio': 0,
                'liquidation_to_borrow_ratio': 0,
                'transactions': []
            }
        
        # Add transaction to wallet's transaction list
        wallets[wallet]['transactions'].append(tx)
        wallets[wallet]['tx_count'] += 1
        
        # Update first and last transaction timestamps
        timestamp = tx.get('timestamp', 0)
        wallets[wallet]['first_tx_timestamp'] = min(wallets[wallet]['first_tx_timestamp'], timestamp)
        wallets[wallet]['last_tx_timestamp'] = max(wallets[wallet]['last_tx_timestamp'], timestamp)
        
        # Extract action data
        action = tx.get('action', '').lower()
        action_data = tx.get('actionData', {})
        amount = float(action_data.get('amount', 0)) / 1e9  # Convert to standard units
        asset_symbol = action_data.get('assetSymbol', '')
        
        if asset_symbol:
            wallets[wallet]['unique_assets'].add(asset_symbol)
        
        # Update action-specific metrics
        if action == 'deposit':
            wallets[wallet]['deposit_count'] += 1
            wallets[wallet]['deposit_amount_total'] += amount
        elif action == 'borrow':
            wallets[wallet]['borrow_count'] += 1
            wallets[wallet]['borrow_amount_total'] += amount
        elif action == 'repay':
            wallets[wallet]['repay_count'] += 1
            wallets[wallet]['repay_amount_total'] += amount
        elif action == 'redeemunderlying':
            wallets[wallet]['redeem_count'] += 1
            wallets[wallet]['redeem_amount_total'] += amount
        elif action == 'liquidationcall':
            wallets[wallet]['liquidation_count'] += 1
            wallets[wallet]['is_liquidated'] = True
            # For liquidation, use collateral amount if available
            collateral_amount = float(action_data.get('collateralAmount', 0)) / 1e18
            principal_amount = float(action_data.get('principalAmount', 0)) / 1e9
            liquidation_amount = collateral_amount if collateral_amount > 0 else principal_amount
            wallets[wallet]['liquidation_amount_total'] += liquidation_amount
    
    # Calculate derived metrics
    for wallet, data in wallets.items():
        # Calculate averages
        data['deposit_amount_avg'] = data['deposit_amount_total'] / data['deposit_count'] if data['deposit_count'] > 0 else 0
        data['borrow_amount_avg'] = data['borrow_amount_total'] / data['borrow_count'] if data['borrow_count'] > 0 else 0
        data['repay_amount_avg'] = data['repay_amount_total'] / data['repay_count'] if data['repay_count'] > 0 else 0
        data['redeem_amount_avg'] = data['redeem_amount_total'] / data['redeem_count'] if data['redeem_count'] > 0 else 0
        data['liquidation_amount_avg'] = data['liquidation_amount_total'] / data['liquidation_count'] if data['liquidation_count'] > 0 else 0
        
        # Calculate ratios
        data['repay_to_borrow_ratio'] = data['repay_amount_total'] / data['borrow_amount_total'] if data['borrow_amount_total'] > 0 else 0
        data['deposit_to_borrow_ratio'] = data['deposit_amount_total'] / data['borrow_amount_total'] if data['borrow_amount_total'] > 0 else 0
        data['liquidation_to_borrow_ratio'] = data['liquidation_amount_total'] / data['borrow_amount_total'] if data['borrow_amount_total'] > 0 else 0
        
        # Calculate account age in days
        data['account_age_days'] = (data['last_tx_timestamp'] - data['first_tx_timestamp']) / (60 * 60 * 24) if data['first_tx_timestamp'] < float('inf') else 0
        
        # Calculate transaction frequency (transactions per day)
        data['tx_frequency'] = data['tx_count'] / data['account_age_days'] if data['account_age_days'] > 0 else 0
        
        # Convert unique assets set to count
        data['unique_assets_count'] = len(data['unique_assets'])
        del data['unique_assets']
        
        # Remove transaction list to save memory
        del data['transactions']
    
    return wallets

def calculate_credit_scores(wallets_features):
    """Calculate credit scores for each wallet"""
    # Convert wallet features to DataFrame
    df = pd.DataFrame.from_dict(wallets_features, orient='index')
    
    # Define feature columns (excluding non-numeric columns)
    feature_columns = [
        'deposit_count', 'deposit_amount_total', 'deposit_amount_avg',
        'borrow_count', 'borrow_amount_total', 'borrow_amount_avg',
        'repay_count', 'repay_amount_total', 'repay_amount_avg',
        'redeem_count', 'redeem_amount_total', 'redeem_amount_avg',
        'liquidation_count', 'liquidation_amount_total', 'liquidation_amount_avg',
        'tx_count', 'repay_to_borrow_ratio', 'deposit_to_borrow_ratio',
        'liquidation_to_borrow_ratio', 'account_age_days', 'tx_frequency',
        'unique_assets_count'
    ]
    
    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Create a scoring model based on key risk indicators
    
    # 1. Base score: Start with 500 points (middle of 0-1000 range)
    df['credit_score'] = 500
    
    # 2. Repayment behavior: +200 points for high repay-to-borrow ratio
    df['credit_score'] += df['repay_to_borrow_ratio'].clip(0, 1) * 200
    
    # 3. Collateralization: +100 points for high deposit-to-borrow ratio
    df['credit_score'] += np.minimum(df['deposit_to_borrow_ratio'] / 2, 1) * 100
    
    # 4. Account history: +100 points for longer account age (max 100 days)
    df['credit_score'] += np.minimum(df['account_age_days'] / 100, 1) * 100
    
    # 5. Transaction diversity: +50 points for using multiple assets
    df['credit_score'] += np.minimum(df['unique_assets_count'] / 5, 1) * 50
    
    # 6. Activity level: +50 points for moderate activity (penalize both too low and too high)
    activity_score = 1 - np.abs(df['tx_frequency'] - 5) / 10
    activity_score = activity_score.clip(0, 1)
    df['credit_score'] += activity_score * 50
    
    # 7. Liquidation penalty: -300 points for being liquidated
    df.loc[df['is_liquidated'], 'credit_score'] -= 300
    
    # 8. Liquidation severity: Additional penalty based on liquidation amount ratio
    df['credit_score'] -= df['liquidation_to_borrow_ratio'].clip(0, 1) * 200
    
    # Ensure scores are within 0-1000 range
    df['credit_score'] = df['credit_score'].clip(0, 1000).round().astype(int)
    
    return df[['credit_score']]

def main():
    # Load transactions
    transactions = load_transactions()
    if not transactions:
        print("No transactions found. Exiting.")
        return
    
    # Extract features
    print("Extracting features...")
    wallet_features = extract_features(transactions)
    print(f"Extracted features for {len(wallet_features)} wallets")
    
    # Calculate credit scores
    print("Calculating credit scores...")
    credit_scores = create_risk_categories(calculate_credit_scores(wallet_features))
    
    # Save results
    output_file = 'wallet_credit_scores.csv'
    credit_scores.to_csv(output_file)
    print(f"Credit scores saved to {output_file}")
    
    # Display some statistics
    print(f"\nCredit Score Statistics:")
    print(f"Average score: {credit_scores['credit_score'].mean():.2f}")
    print(f"Median score: {credit_scores['credit_score'].median()}")
    print(f"Min score: {credit_scores['credit_score'].min()}")
    print(f"Max score: {credit_scores['credit_score'].max()}")
    
    # Display sample of scores
    print(f"\nSample of wallet credit scores:")
    print(credit_scores.head(10))

if __name__ == "__main__":
    main()