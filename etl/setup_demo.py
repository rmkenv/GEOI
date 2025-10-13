#!/usr/bin/env python3
"""
Configuration example and demo data generator for the Geospatial Finance Intelligence System
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import random
import json

# Example .env file content
env_content = '''# Geospatial Finance Intelligence System - Environment Variables

# SEC API Configuration
SEC_API_KEY=your_sec_api_key_from_sec_api_io

# Anthropic AI Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_from_console_anthropic_com

# Data Configuration
DATA_DIR=./geospatial_finance_data
LOOKBACK_DAYS=7
MAX_RECORDS_PER_QUERY=100

# Rate Limiting
REQUEST_DELAY=0.1
MAX_RETRIES=3

# Dashboard Configuration
STREAMLIT_PORT=8501
DEBUG_MODE=False
'''

def load_geoi_tickers(snapshot_url=None):
    """Load tickers from latest GEOI parquet snapshot"""
    snapshot_url = snapshot_url or "https://raw.githubusercontent.com/rmkenv/GEOI/main/snapshots/2025/snapshot_2025-10-09.parquet"
    df = pd.read_parquet(snapshot_url)
    tickers = df['ticker'].dropna().unique().tolist()
    return tickers

def create_demo_data():
    """Create demo data for testing the system without API calls"""

    # Ensure data directory exists
    data_dir = Path('./geospatial_finance_data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load GEOI tickers
    tickers = load_geoi_tickers()
    # Create simple company dict list with default names/sectors
    companies = [{'ticker': t, 'name': t, 'sector': 'Geospatial'} for t in tickers]

    # Generate sample data
    demo_records = []

    # SEC Filings
    for i in range(50):
        company = random.choice(companies)
        filing_date = datetime.now() - timedelta(days=random.randint(0, 30))

        demo_records.append({
            'ticker': company['ticker'],
            'companyName': company['name'],
            'cik': f"{random.randint(1000000, 9999999)}",
            'formType': random.choice(['10-K', '10-Q', '8-K', 'S-1', 'DEF 14A']),
            'filedAt': filing_date,
            'accessionNo': f"0001{random.randint(100000, 999999)}-{random.randint(10, 99)}-{random.randint(100000, 999999)}",
            'linkToFilingDetails': f"https://www.sec.gov/filing/example-{i}",
            'items': random.choice(['Item 1.01', 'Item 2.02', 'Item 8.01', '']),
            'description': f"Form {random.choice(['10-K', '10-Q', '8-K'])} filing mentioning geospatial technology",
            'source': 'SEC_EDGAR',
            'category': 'filing'
        })

    # Insider Trading Data
    for i in range(30):
        company = random.choice(companies)
        trade_date = datetime.now() - timedelta(days=random.randint(0, 60))

        demo_records.append({
            'filedAt': trade_date,
            'issuerTicker': company['ticker'],
            'issuerName': company['name'],
            'reportingPerson': random.choice([
                'John Smith, CEO', 'Sarah Johnson, CFO', 'Michael Brown, CTO',
                'Lisa Davis, Director', 'Robert Wilson, VP Sales'
            ]),
            'transactionCode': random.choice(['P', 'S', 'A', 'D']),  # Purchase, Sale, Award, Disposition
            'shares': random.randint(1000, 50000),
            'pricePerShare': round(random.uniform(10, 200), 2),
            'totalValue': 0,  # Will calculate below
            'source': 'SEC_FORM4',
            'category': 'insider_trading'
        })
        # Calculate total value
        demo_records[-1]['totalValue'] = demo_records[-1]['shares'] * demo_records[-1]['pricePerShare']

    # 13F Holdings Data
    for i in range(40):
        company = random.choice(companies)
        filing_date = datetime.now() - timedelta(days=random.randint(0, 90))

        demo_records.append({
            'filedAt': filing_date,
            'cik': f"{random.randint(1000000, 9999999)}",
            'nameOfIssuer': company['name'],
            'ticker': company['ticker'],
            'cusip': f"{random.randint(100000, 999999)}{random.randint(10, 99)}{random.randint(1, 9)}",
            'value': random.randint(1000000, 100000000),  # $1M to $100M
            'shrsOrPrnAmt': random.randint(10000, 1000000),
            'investmentDiscretion': random.choice(['SOLE', 'SHARED', 'NONE']),
            'source': 'SEC_13F',
            'category': 'institutional_holding'
        })

    # Lobbying Data (placeholder)
    lobbying_clients = [
        'Esri Inc', 'Maxar Technologies', 'Planet Labs', 'Trimble Inc',
        'Geospatial Intelligence Association', 'National Geospatial Association'
    ]

    for i in range(20):
        quarter_start = datetime.now().replace(month=((datetime.now().month-1)//3)*3+1, day=1)

        demo_records.append({
            'client': random.choice(lobbying_clients),
            'registrant': random.choice([
                'Capitol Counsel LLC', 'K&L Gates LLP', 'Akin Gump Strauss',
                'Williams & Jensen', 'Brownstein Hyatt'
            ]),
            'issue': random.choice([
                'Geospatial mapping technologies and privacy',
                'Satellite imagery export controls',
                'Earth observation data policies',
                'GPS and location services regulation',
                'Government geospatial data sharing'
            ]),
            'amount': random.randint(20000, 200000),
            'year': datetime.now().year,
            'quarter': f'Q{((datetime.now().month-1)//3)+1}',
            'filedAt': quarter_start,
            'source': 'LOBBYING_DISCLOSURE',
            'category': 'lobbying_activity'
        })

    # Congressional Trading (placeholder)
    for i in range(15):
        trade_date = datetime.now() - timedelta(days=random.randint(0, 180))

        demo_records.append({
            'representative': random.choice([
                'Rep. John Doe (R-TX)', 'Sen. Jane Smith (D-CA)', 
                'Rep. Michael Johnson (R-FL)', 'Sen. Sarah Williams (D-NY)'
            ]),
            'transaction_type': random.choice(['Purchase', 'Sale']),
            'ticker': random.choice([c['ticker'] for c in companies]),
            'amount_range': random.choice([
                '$1,001 - $15,000', '$15,001 - $50,000', 
                '$50,001 - $100,000', '$100,001 - $250,000'
            ]),
            'transaction_date': trade_date,
            'disclosure_date': trade_date + timedelta(days=random.randint(1, 45)),
            'filedAt': trade_date + timedelta(days=random.randint(1, 45)),
            'source': 'CONGRESSIONAL_TRADING',
            'category': 'political_trading'
        })

    # Company Metrics
    for company in companies:
        demo_records.append({
            'ticker': company['ticker'],
            'company_name': company['name'],
            'sector': company['sector'],
            'market_cap': random.randint(500000000, 20000000000),  # $500M to $20B
            'last_updated': datetime.now(),
            'filedAt': datetime.now(),
            'source': 'COMPANY_FUNDAMENTALS',
            'category': 'company_metrics'
        })

    # Create DataFrame and add common fields
    df = pd.DataFrame(demo_records)
    df['data_collection_timestamp'] = datetime.now()
    df['week'] = pd.to_datetime(df['filedAt']).dt.isocalendar().week
    df['year'] = pd.to_datetime(df['filedAt']).dt.year

    # Save to parquet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"geospatial_finance_demo_{timestamp}.parquet"

    df.to_parquet(output_file, compression='snappy', index=False)

    print(f"✅ Created demo data: {output_file}")
    print(f"📊 Generated {len(df)} records across {df['category'].nunique()} categories")
    print("\nCategory breakdown:")
    print(df['category'].value_counts().to_string())

    return output_file

def main():
    """Create example configuration files and demo data"""

    # Create .env example
    with open('.env.example', 'w') as f:
        f.write(env_content)

    print("✅ Created .env.example file")

    # Generate demo data
    output_file = create_demo_data()

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\n1. Copy .env.example to .env and add your API keys")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Test with demo data:")
    print(f"   python geospatial_finance_cli.py status")
    print(f"   streamlit run geospatial_finance_dashboard.py")
    print("\n4. Run real ETL pipeline:")
    print("   python geospatial_finance_etl.py")

if __name__ == "__main__":
    main()
