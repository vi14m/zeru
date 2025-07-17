# DeFi Wallet Credit Scoring Model

This project implements a machine learning model that assigns credit scores to cryptocurrency wallets based on their transaction history with the Aave V2 protocol. The scores range from 0 to 1000, with higher scores indicating more reliable and responsible usage patterns.

## Features

The credit scoring model analyzes the following aspects of wallet behavior:

1. **Repayment Behavior**: How consistently and completely loans are repaid
2. **Collateralization Ratio**: The ratio of deposits to borrowings
3. **Account History**: The age and consistency of the account
4. **Transaction Diversity**: The variety of assets used
5. **Activity Level**: The frequency and pattern of transactions
6. **Liquidation History**: Whether the wallet has experienced liquidations

## How Scores Are Calculated

The model uses a rule-based approach with the following components:

- Base score: 500 points (middle of the range)
- Repayment behavior: Up to +200 points for high repay-to-borrow ratio
- Collateralization: Up to +100 points for high deposit-to-borrow ratio
- Account history: Up to +100 points for longer account age
- Transaction diversity: Up to +50 points for using multiple assets
- Activity level: Up to +50 points for moderate activity
- Liquidation penalty: -300 points for being liquidated
- Liquidation severity: Up to -200 additional points based on liquidation amount

## Usage

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn

### Installation

```bash
pip install pandas numpy scikit-learn
```

### Running the Model

1. Place your transaction data in a file named `user-wallet-transactions.json` in the same directory as the script
2. Run the script:

```bash
python credit_score_model.py
```

3. The results will be saved to `wallet_credit_scores.csv`

## Input Data Format

The input JSON file should contain an array of transaction objects with the following structure:

```json
{
  "userWallet": "0x...",
  "network": "polygon",
  "protocol": "aave_v2",
  "timestamp": 1629178166,
  "action": "deposit",
  "actionData": {
    "amount": "2000000000",
    "assetSymbol": "USDC",
    "userId": "0x..."
  }
}
```

Supported action types: `deposit`, `borrow`, `repay`, `redeemunderlying`, and `liquidationcall`.

## Output

The script generates a CSV file with wallet addresses and their corresponding credit scores (0-1000).

## Future Improvements

- Implement more sophisticated machine learning models (Random Forest, XGBoost)
- Add time-based features to capture behavioral changes
- Incorporate external data sources for enhanced risk assessment
- Implement model validation and performance metrics