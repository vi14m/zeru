# Credit Scoring for Cryptocurrency Wallets

## Overview

This system analyzes transaction data from the Aave V2 protocol to assign credit scores to cryptocurrency wallets. The scores range from 0 to 1000, with higher scores indicating more reliable and responsible usage patterns. The system provides both a rule-based model and an advanced machine learning model for credit scoring.

## System Architecture

### Data Flow

1.  **Input**: JSON file containing transaction records from Aave V2 protocol
2.  **Processing**: Feature extraction and credit score calculation
3.  **Output**: CSV file with wallet addresses and credit scores, visualizations

### Components

1.  **Analyze Transactions**: `analyze_transactions.py`
2.  **Basic Model**: `credit_score_model.py`
3.  **Advanced Model**: `advanced_credit_model.py`
4.  **Visualization**: `visualize_results.py`
5.  **Pipeline Runner**: `run_credit_scoring.bat`

## Feature Engineering

The system extracts the following features from transaction data:

### Transaction Counts
- Number of deposits
- Number of borrows
- Number of repayments
- Number of redemptions
- Number of liquidations

### Amount Metrics
- Total deposit amount
- Total borrow amount
- Total repay amount
- Total redeem amount
- Total liquidation amount
- Average amounts for each action type

### Ratio Metrics
- Repay-to-borrow ratio
- Deposit-to-borrow ratio
- Liquidation-to-borrow ratio

### Time-based Metrics
- Account age (days)
- Transaction frequency
- Average time between transactions
- Time since last transaction

### Diversity Metrics
- Number of unique assets used
- Asset concentration

### Risk Indicators
- Liquidation flag
- Maximum borrow amount
- Maximum deposit amount

## Credit Scoring Models

### Rule-based Model

The rule-based model calculates credit scores using a weighted combination of key risk indicators:

1.  **Base Score**: 500 points (middle of 0-1000 range)
2.  **Repayment Behavior**: +200 points for high repay-to-borrow ratio
3.  **Collateralization**: +100 points for high deposit-to-borrow ratio
4.  **Account History**: +100 points for longer account age
5.  **Transaction Diversity**: +50 points for using multiple assets
6.  **Activity Level**: +50 points for moderate activity
7.  **Liquidation Penalty**: -300 points for being liquidated
8.  **Liquidation Severity**: Additional penalty based on liquidation amount ratio

### Machine Learning Model

The advanced model uses ensemble learning techniques to predict credit scores:

1.  **Data Preparation**:
    - Feature extraction from transaction data
    - Creation of synthetic labels for training
    - Feature scaling

2.  **Model Training**:
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - KNN
    - XGBoost
    - Model selection based on performance metrics

3.  **Evaluation Metrics**:
    - Mean Squared Error (MSE)
    - R-squared (R²)

4.  **Feature Importance Analysis**:
    - Identification of most predictive features
    - Visualization of feature importance

## Risk Categories

Based on credit scores, wallets are classified into risk categories:

-   **High Risk** (0-250): Wallets with severe issues, such as multiple liquidations or very low repayment rates
-   **Medium-High Risk** (250-500): Wallets with concerning patterns, such as low collateralization or inconsistent repayments
-   **Medium-Low Risk** (500-750): Wallets with generally good behavior but some minor issues
-   **Low Risk** (750-1000): Wallets with excellent repayment history, good collateralization, and no liquidations

## Usage Instructions

### Prerequisites

-   Python 3.7+
-   Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

1.  Execute the batch script:

```bash
run_credit_scoring.bat
```

2.  Select the model type:
    -   Option 1: Basic rule-based model
    -   Option 2: Advanced machine learning model

3.  Review the results in the output files

## Output Files

### Basic Model
-   `wallet_credit_scores.csv`: Raw credit scores
-   `score_distribution.png`: Visualization of score distribution
-   `risk_categories.png`: Visualization of risk categories

### Advanced Model
-   `wallet_credit_scores_ml.csv`: Raw credit scores from ML model
-   `credit_score_model.pkl`: Trained ML model
-   `feature_scaler.pkl`: Feature scaler for the ML model
-   `feature_importance.png`: Visualization of feature importance
-   `ml_score_distribution.png`: Visualization of score distribution

## Potential Applications

1.  **Risk Assessment**: Evaluate wallet creditworthiness for undercollateralized loans
2.  **User Segmentation**: Identify high-value, low-risk users for targeted offerings
3.  **Fraud Detection**: Flag suspicious activity patterns
4.  **Protocol Optimization**: Adjust parameters based on user risk profiles
5.  **Incentive Design**: Create reward systems for responsible protocol usage

## Limitations and Future Work

### Current Limitations

-   Relies solely on on-chain transaction data
-   Limited historical context for new wallets
-   No cross-protocol data integration

### Future Improvements

-   Incorporate cross-chain activity data
-   Add time-series analysis for behavioral changes
-   Implement anomaly detection for suspicious activities
-   Develop real-time scoring capabilities
-   Create API for integration with other DeFi applications

## Technical Details

### Data Normalization

-   Amount values are normalized to standard units (dividing by 10^9 for most tokens)
-   Categorical features are one-hot encoded
-   Numerical features are standardized using StandardScaler

### Model Persistence

The trained machine learning model and feature scaler are saved using joblib for future use without retraining:

```python
joblib.dump(model, 'credit_score_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
```

### Performance Considerations

-   The system is designed to handle large transaction datasets
-   Memory optimization techniques are used for feature extraction
-   Batch processing is implemented for scalability