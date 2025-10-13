
etl_against_geoi_code = """
import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GEOILoader:
    def __init__(self, snapshot_url=None):
        self.snapshot_url = snapshot_url or "https://raw.githubusercontent.com/rmkenv/GEOI/main/snapshots/2025/snapshot_2025-10-09.parquet"

    def load_tickers(self):
        logger.info(f"Loading GEOI snapshot from {self.snapshot_url}")
        df = pd.read_parquet(self.snapshot_url)
        tickers = df['ticker'].dropna().unique().tolist()
        logger.info(f"Loaded {len(tickers)} unique tickers")
        return tickers


class SECDataCollector:
    def __init__(self, sec_api_key, rate_limit_delay=0.2):
        self.api_key = sec_api_key
        self.base_url = "https://api.sec-api.io"
        self.headers = {
            'Authorization': f'Bearer {sec_api_key}',
            'User-Agent': 'GEOI-ETL-Agent/1.0'
        }
        self.rate_limit_delay = rate_limit_delay

    def query_sec_filings(self, ticker, days_back=7):
        logger.info(f"Querying SEC filings for {ticker}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        query = (f'ticker:"{ticker}" AND filedAt:[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]')

        params = {
            'query': query,
            'from': 0,
            'size': 100,
            'sort': [{'filedAt': {'order': 'desc'}}]
        }
        
        try:
            response = requests.get(f'{self.base_url}/query', headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            filings = data.get('filings', [])
        except Exception as e:
            logger.error(f'Error querying SEC filings for {ticker}: {e}')
            filings = []

        time.sleep(self.rate_limit_delay)
        return filings

    def query_13f_holdings(self, ticker, days_back=90):
        logger.info(f"Querying 13F holdings for {ticker}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        query = (f'ticker:"{ticker}" AND formType:"13F-HR" AND filedAt:[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]')

        params = {
            'query': query,
            'from': 0,
            'size': 50,
            'sort': [{'filedAt': {'order': 'desc'}}]
        }

        try:
            response = requests.get(f'{self.base_url}/form-13f-holdings', headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            holdings = data.get('data', [])
        except Exception as e:
            logger.error(f'Error querying 13F holdings for {ticker}: {e}')
            holdings = []

        time.sleep(self.rate_limit_delay)
        return holdings

    def query_insider_trades(self, ticker, days_back=30):
        logger.info(f"Querying insider trades for {ticker}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        query = (f'ticker:"{ticker}" AND formType:"4" AND filedAt:[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]')

        params = {
            'query': query,
            'from': 0,
            'size': 50,
            'sort': [{'filedAt': {'order': 'desc'}}]
        }

        try:
            response = requests.get(f'{self.base_url}/insider-trading', headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            trades = data.get('transactions', [])
        except Exception as e:
            logger.error(f'Error querying insider trades for {ticker}: {e}')
            trades = []

        time.sleep(self.rate_limit_delay)
        return trades


def flatten_filings(filings, ticker):
    results = []
    for f in filings:
        results.append({
            'ticker': ticker,
            'formType': f.get('formType'),
            'filedAt': f.get('filedAt'),
            'description': f.get('description'),
            'accessionNo': f.get('accessionNo'),
            'linkToDetails': f.get('linkToFilingDetails'),
            'source': 'SEC_FILINGS'
        })
    return results

def flatten_13f(holdings, ticker):
    results = []
    for h in holdings:
        holdings_list = h.get('holdings', [])
        for item in holdings_list:
            results.append({
                'ticker': ticker,
                'filedAt': h.get('filedAt'),
                'nameOfIssuer': item.get('nameOfIssuer'),
                'value': item.get('value'),
                'shares': item.get('shrsOrPrnAmt', {}).get('sshPrnamt'),
                'cusip': item.get('cusip'),
                'source': 'SEC_13F'
            })
    return results

def flatten_trades(trades, ticker):
    results = []
    for t in trades:
        results.append({
            'ticker': ticker,
            'filedAt': t.get('filedAt'),
            'reportingPerson': t.get('reportingOwner', {}).get('name'),
            'transactionCode': t.get('transactionCode'),
            'shares': t.get('sharesTransacted'),
            'pricePerShare': t.get('pricePerShare'),
            'totalValue': t.get('totalValue'),
            'source': 'SEC_FORM4'
        })
    return results


def run_etl_on_geoi():
    loader = GEOILoader()
    tickers = loader.load_tickers()

    api_key = os.getenv('SEC_API_KEY')
    if not api_key:
        logger.error('Please set the SEC_API_KEY environment variable')
        return

    collector = SECDataCollector(api_key)

    all_data = []
    for ticker in tickers:
        filings = collector.query_sec_filings(ticker)
        all_data.extend(flatten_filings(filings, ticker))

        holdings = collector.query_13f_holdings(ticker)
        all_data.extend(flatten_13f(holdings, ticker))

        trades = collector.query_insider_trades(ticker)
        all_data.extend(flatten_trades(trades, ticker))

    if not all_data:
        logger.warning('No data collected from SEC API')
        return

    df = pd.DataFrame(all_data)
    df['filedAt'] = pd.to_datetime(df['filedAt'], errors='coerce')
    output_dir = Path('./geospatial_finance_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = output_dir / f'geospatial_finance_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
    df.to_parquet(filename, index=False)
    logger.info(f'Saved combined data to {filename}')

if __name__ == '__main__':
    run_etl_on_geoi()

"""

with open('etl_geoi.py', 'w') as f:
    f.write(etl_against_geoi_code)

print("✅ Created ETL script to query SEC for GEOI tickers: etl_geoi.py")
print("Run this script after setting SEC_API_KEY environment variable")
print("For example: export SEC_API_KEY='your_sec_api_key_here'")
print("Then: python etl_geoi.py")
