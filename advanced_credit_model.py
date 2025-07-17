import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import xgboost as xgb
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
                # Basic transaction counts
                'deposit_count': 0,
                'borrow_count': 0,
                'repay_count': 0,
                'redeem_count': 0,
                'liquidation_count': 0,
                
                # Amount totals
                'deposit_amount_total': 0,
                'borrow_amount_total': 0,
                'repay_amount_total': 0,
                'redeem_amount_total': 0,
                'liquidation_amount_total': 0,
                
                # Amount averages
                'deposit_amount_avg': 0,
                'borrow_amount_avg': 0,
                'repay_amount_avg': 0,
                'redeem_amount_avg': 0,
                'liquidation_amount_avg': 0,
                
                # Time-based metrics
                'first_tx_timestamp': float('inf'),
                'last_tx_timestamp': 0,
                'avg_time_between_tx': 0,
                'time_since_last_tx': 0,
                
                # Asset diversity
                'unique_assets': set(),
                
                # Transaction patterns
                'tx_count': 0,
                'deposit_to_borrow_tx_ratio': 0,
                'repay_to_borrow_tx_ratio': 0,
                
                # Risk indicators
                'is_liquidated': False,
                'liquidation_to_borrow_ratio': 0,
                'max_borrow_amount': 0,
                'max_deposit_amount': 0,
                
                # Temporal patterns
                'timestamps': [],
                'transactions': []
            }
        
        # Add transaction to wallet's transaction list
        wallets[wallet]['transactions'].append(tx)
        wallets[wallet]['timestamps'].append(tx.get('timestamp', 0))
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
            wallets[wallet]['max_deposit_amount'] = max(wallets[wallet]['max_deposit_amount'], amount)
        elif action == 'borrow':
            wallets[wallet]['borrow_count'] += 1
            wallets[wallet]['borrow_amount_total'] += amount
            wallets[wallet]['max_borrow_amount'] = max(wallets[wallet]['max_borrow_amount'], amount)
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
        data['repay_to_borrow_tx_ratio'] = data['repay_count'] / data['borrow_count'] if data['borrow_count'] > 0 else 0
        data['deposit_to_borrow_tx_ratio'] = data['deposit_count'] / data['borrow_count'] if data['borrow_count'] > 0 else 0
        
        # Calculate account age in days
        data['account_age_days'] = (data['last_tx_timestamp'] - data['first_tx_timestamp']) / (60 * 60 * 24) if data['first_tx_timestamp'] < float('inf') else 0
        
        # Calculate transaction frequency (transactions per day)
        data['tx_frequency'] = data['tx_count'] / data['account_age_days'] if data['account_age_days'] > 0 else 0
        
        # Calculate time between transactions
        if len(data['timestamps']) > 1:
            sorted_timestamps = sorted(data['timestamps'])
            time_diffs = [sorted_timestamps[i+1] - sorted_timestamps[i] for i in range(len(sorted_timestamps)-1)]
            data['avg_time_between_tx'] = sum(time_diffs) / len(time_diffs) / (60 * 60)  # in hours
            current_time = datetime.now().timestamp()
            data['time_since_last_tx'] = (current_time - data['last_tx_timestamp']) / (60 * 60 * 24)  # in days
        
        # Calculate volatility of transaction frequency
        if len(data['timestamps']) > 2:
            sorted_timestamps = sorted(data['timestamps'])
            time_diffs = [sorted_timestamps[i+1] - sorted_timestamps[i] for i in range(len(sorted_timestamps)-1)]
            data['tx_time_volatility'] = np.std(time_diffs) / (60 * 60)  # in hours
        else:
            data['tx_time_volatility'] = 0
        
        # Convert unique assets set to count
        data['unique_assets_count'] = len(data['unique_assets'])
        del data['unique_assets']
        
        # Calculate asset concentration (if one asset dominates)
        asset_counts = {}
        for tx in data['transactions']:
            asset = tx.get('actionData', {}).get('assetSymbol', '')
            if asset:
                asset_counts[asset] = asset_counts.get(asset, 0) + 1
        
        if asset_counts and data['tx_count'] > 0:
            max_asset_count = max(asset_counts.values())
            data['asset_concentration'] = max_asset_count / data['tx_count']
        else:
            data['asset_concentration'] = 0
        
        # Remove transaction list and timestamps to save memory
        del data['transactions']
        del data['timestamps']
    
    return wallets

def create_training_data(wallets_features):
    """Create training data with synthetic labels for model training"""
    # Convert wallet features to DataFrame
    df = pd.DataFrame.from_dict(wallets_features, orient='index')
    
    # Create synthetic labels based on known risk factors
    # This is a simplified approach - in a real scenario, you would use historical default data
    
    # Start with a base score
    df['synthetic_score'] = 500
    
    # Positive factors
    df['synthetic_score'] += df['repay_to_borrow_ratio'].clip(0, 1) * 200
    df['synthetic_score'] += np.minimum(df['deposit_to_borrow_ratio'] / 2, 1) * 100
    df['synthetic_score'] += np.minimum(df['account_age_days'] / 100, 1) * 100
    df['synthetic_score'] += np.minimum(df['unique_assets_count'] / 5, 1) * 50
    
    # Negative factors
    df.loc[df['is_liquidated'], 'synthetic_score'] -= 300
    df['synthetic_score'] -= df['liquidation_to_borrow_ratio'].clip(0, 1) * 200
    
    # Add some random noise to make it more realistic
    df['synthetic_score'] += np.random.normal(0, 50, size=len(df))
    
    # Ensure scores are within 0-1000 range
    df['synthetic_score'] = df['synthetic_score'].clip(0, 1000)
    
    return df

def train_model(df):
    """Train a machine learning model to predict credit scores"""
    # Define feature columns (excluding non-numeric columns and the target)
    feature_columns = [
        'deposit_count', 'borrow_count', 'repay_count', 'redeem_count', 'liquidation_count',
        'deposit_amount_total', 'borrow_amount_total', 'repay_amount_total', 'redeem_amount_total',
        'deposit_amount_avg', 'borrow_amount_avg', 'repay_amount_avg', 'redeem_amount_avg',
        'account_age_days', 'tx_frequency', 'unique_assets_count', 'is_liquidated',
        'repay_to_borrow_ratio', 'deposit_to_borrow_ratio', 'liquidation_to_borrow_ratio',
        'repay_to_borrow_tx_ratio', 'deposit_to_borrow_tx_ratio', 'tx_count',
        'max_borrow_amount', 'max_deposit_amount', 'asset_concentration'
    ]
    
    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Split data into features and target
    X = df[feature_columns]
    y = df['synthetic_score']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store model results
    model_results = {}
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    rf_pred = rf_model.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    model_results['Random Forest'] = {'model': rf_model, 'r2': rf_r2, 'mse': rf_mse}
    
    print(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")
    
    # Train a Gradient Boosting model
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    gb_pred = gb_model.predict(X_test_scaled)
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    model_results['Gradient Boosting'] = {'model': gb_model, 'r2': gb_r2, 'mse': gb_mse}
    
    print(f"Gradient Boosting - MSE: {gb_mse:.2f}, R²: {gb_r2:.2f}")
    
    # Train a KNN model
    print("Training KNN model...")
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    knn_pred = knn_model.predict(X_test_scaled)
    knn_mse = mean_squared_error(y_test, knn_pred)
    knn_r2 = r2_score(y_test, knn_pred)
    model_results['KNN'] = {'model': knn_model, 'r2': knn_r2, 'mse': knn_mse}
    
    print(f"KNN - MSE: {knn_mse:.2f}, R²: {knn_r2:.2f}")
    
    # Train an XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    model_results['XGBoost'] = {'model': xgb_model, 'r2': xgb_r2, 'mse': xgb_mse}
    
    print(f"XGBoost - MSE: {xgb_mse:.2f}, R²: {xgb_r2:.2f}")
    
    # Choose the best model based on R² score
    best_model_name = max(model_results, key=lambda k: model_results[k]['r2'])
    best_model = model_results[best_model_name]['model']
    print(f"\nUsing {best_model_name} model for predictions (best R² score)")
    
    # Feature importance (if the model supports it)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Feature Importance for Credit Score Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    else:
        print("\nSelected model does not support feature importance visualization")
    
    # Save the model and scaler
    joblib.dump(best_model, 'credit_score_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    return best_model, scaler, feature_columns

def predict_credit_scores(wallets_features, model, scaler, feature_columns):
    """Predict credit scores for each wallet"""
    # Convert wallet features to DataFrame
    df = pd.DataFrame.from_dict(wallets_features, orient='index')
    
    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Scale features
    X = scaler.transform(df[feature_columns])
    
    # Predict scores
    scores = model.predict(X)
    
    # Ensure scores are within 0-1000 range and round to integers
    scores = np.clip(scores, 0, 1000).round().astype(int)
    
    # Create DataFrame with scores
    scores_df = pd.DataFrame({
        'credit_score': scores
    }, index=df.index)
    
    return scores_df

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
    
    # Create training data
    print("Creating training data...")
    training_df = create_training_data(wallet_features)
    
    # Train model
    print("Training model...")
    model, scaler, feature_columns = train_model(training_df)
    
    # Predict credit scores
    print("Predicting credit scores...")
    credit_scores = predict_credit_scores(wallet_features, model, scaler, feature_columns)
    
    # Save results
    # Create and add risk categories
    credit_scores = create_risk_categories(credit_scores)

    output_file = 'wallet_credit_scores_ml.csv'
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
    
    # Visualize score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(credit_scores['credit_score'], kde=True, bins=20)
    plt.axvline(x=250, color='r', linestyle='--', alpha=0.7, label='High Risk (0-250)')
    plt.axvline(x=500, color='y', linestyle='--', alpha=0.7, label='Medium Risk (250-500)')
    plt.axvline(x=750, color='g', linestyle='--', alpha=0.7, label='Low Risk (500-750)')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.title('Distribution of Wallet Credit Scores (Machine Learning Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ml_score_distribution.png')
    plt.close()

if __name__ == "__main__":
    main()