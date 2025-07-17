import json
import os
from collections import Counter
from datetime import datetime

# Define the path to the JSON file
JSON_FILE_PATH = 'user-wallet-transactions.json'

def load_transactions():
    """Load transactions from JSON file"""
    try:
        with open(JSON_FILE_PATH, 'r') as file:
            transactions = json.load(file)
        print(f"Loaded {len(transactions)} transactions from {JSON_FILE_PATH}")
        return transactions
    except FileNotFoundError:
        print(f"Error: The file {JSON_FILE_PATH} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {JSON_FILE_PATH}. Check file format.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading transactions: {e}")
        return []

def analyze_wallet_distribution(transactions):
    """Analyze the distribution of transactions per wallet"""
    wallet_counts = Counter(tx.get('userWallet') for tx in transactions if tx.get('userWallet'))
    num_unique_wallets = len(wallet_counts)
    total_transactions = len(transactions)

    print(f"\n--- Wallet Distribution Analysis ---")
    print(f"Total transactions: {total_transactions}")
    print(f"Number of unique wallets: {num_unique_wallets}")

    if num_unique_wallets > 0:
        avg_tx_per_wallet = total_transactions / num_unique_wallets
        print(f"Average transactions per wallet: {avg_tx_per_wallet:.2f}")

        # Display top 5 wallets by transaction count
        print("Top 5 wallets by transaction count:")
        for wallet, count in wallet_counts.most_common(5):
            print(f"  {wallet}: {count} transactions")

        # Display bottom 5 wallets by transaction count (excluding those with 0 or 1 tx if many)
        if num_unique_wallets > 5:
            print("Bottom 5 wallets by transaction count:")
            for wallet, count in list(wallet_counts.most_common())[-5:]:
                print(f"  {wallet}: {count} transactions")
    else:
        print("No wallet data to analyze.")

def analyze_action_distribution(transactions):
    """Analyze the distribution of different action types"""
    action_counts = Counter(tx.get('action') for tx in transactions if tx.get('action'))

    print(f"\n--- Action Type Distribution ---")
    if action_counts:
        for action, count in action_counts.most_common():
            print(f"  {action}: {count} transactions")
    else:
        print("No action data to analyze.")

def analyze_asset_distribution(transactions):
    """Analyze the distribution of assets involved in transactions"""
    asset_counts = Counter()
    for tx in transactions:
        action_data = tx.get('actionData', {})
        asset_symbol = action_data.get('assetSymbol')
        if asset_symbol:
            asset_counts[asset_symbol] += 1

    print(f"\n--- Asset Distribution ---")
    if asset_counts:
        for asset, count in asset_counts.most_common(10):
            print(f"  {asset}: {count} transactions")
    else:
        print("No asset data to analyze.")

def analyze_temporal_patterns(transactions):
    """Analyze temporal patterns of transactions (e.g., by year, month) """
    timestamps = [tx.get('timestamp') for tx in transactions if tx.get('timestamp')]
    if not timestamps:
        print("\n--- Temporal Patterns ---")
        print("No timestamp data to analyze.")
        return

    # Convert timestamps to datetime objects
    # Assuming timestamps are in seconds since epoch
    dates = []
    for ts in timestamps:
        try:
            dates.append(datetime.fromtimestamp(ts))
        except (TypeError, ValueError):
            # Handle cases where timestamp might be invalid or in milliseconds
            try:
                dates.append(datetime.fromtimestamp(ts / 1000)) # Try milliseconds
            except (TypeError, ValueError):
                continue # Skip invalid timestamps

    if not dates:
        print("\n--- Temporal Patterns ---")
        print("No valid timestamp data to analyze.")
        return

    years = Counter(date.year for date in dates)
    months = Counter(date.strftime('%Y-%m') for date in dates)

    print(f"\n--- Temporal Patterns ---")
    print("Transactions by Year:")
    for year, count in sorted(years.items()):
        print(f"  {year}: {count}")

    print("\nTransactions by Month (Top 10 Most Active):")
    for month, count in months.most_common(10):
        print(f"  {month}: {count}")

    # Calculate transaction rate over time
    min_date = min(dates)
    max_date = max(dates)
    total_days = (max_date - min_date).days

    if total_days > 0:
        tx_per_day = len(transactions) / total_days
        print(f"\nOverall transaction rate: {tx_per_day:.2f} transactions/day")
    else:
        print("Not enough temporal spread to calculate transaction rate.")

def main():
    print("Starting JSON data analysis...")
    transactions = load_transactions()

    if not transactions:
        print("Analysis cannot proceed without transaction data.")
        return

    analyze_wallet_distribution(transactions)
    analyze_action_distribution(transactions)
    analyze_asset_distribution(transactions)
    analyze_temporal_patterns(transactions)

    print("\nJSON data analysis complete.")

if __name__ == "__main__":
    main()