@echo off
echo DeFi Wallet Credit Scoring Pipeline
echo ===================================
echo.

echo Step 1: Checking for Python installation...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.7+ and try again.
    exit /b 1
)

echo Step 2: Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install required packages.
    exit /b 1
)

echo Step 3: Preparing transaction data...
if not exist user-wallet-transactions.json (
    echo Error: No transaction data found. Please provide user-wallet-transactions.json file.
    exit /b 1
) else (
    echo Found existing transaction data.
    echo Analyzing transaction data...
    python analyze_transactions.py
)

echo.
echo Please select a model to run:
echo 1. Basic rule-based credit scoring model
echo 2. Advanced machine learning credit scoring model
echo.

set /p model_choice=Enter your choice (1 or 2): 

if "%model_choice%"=="1" (
    echo.
    echo Step 4: Running basic credit scoring model...
    python credit_score_model.py
    if %errorlevel% neq 0 (
        echo Error: Failed to run credit scoring model.
        exit /b 1
    )
    
    echo Step 5: Visualizing results...
    python visualize_results.py
    if %errorlevel% neq 0 (
        echo Error: Failed to visualize results.
        exit /b 1
    )
    
    echo.
    echo Basic credit scoring pipeline completed successfully!
    echo Results are available in the following files:
    echo - wallet_credit_scores.csv: Raw credit scores
    echo - wallet_risk_categories.csv: Categorized risk levels
    echo - score_distribution.png: Visualization of score distribution
    echo - risk_categories.png: Visualization of risk categories
) else if "%model_choice%"=="2" (
    echo.
    echo Step 4: Running advanced machine learning credit scoring model...
    python advanced_credit_model.py
    if %errorlevel% neq 0 (
        echo Error: Failed to run advanced credit scoring model.
        exit /b 1
    )
    
    echo.
    echo Advanced credit scoring pipeline completed successfully!
    echo Results are available in the following files:
    echo - wallet_credit_scores_ml.csv: Raw credit scores from ML model
    echo - credit_score_model.pkl: Trained ML model
    echo - feature_scaler.pkl: Feature scaler for the ML model
    echo - feature_importance.png: Visualization of feature importance
    echo - ml_score_distribution.png: Visualization of score distribution
) else (
    echo.
    echo Invalid choice. Please run the script again and select 1 or 2.
    exit /b 1
)

pause