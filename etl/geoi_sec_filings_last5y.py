import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USER_AGENT = "YourName YourCompany your.email@example.com"  # ⚠️ CHANGE THIS TO YOUR INFO
OUTPUT_DIR = Path("./geospatial_finance_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Use your newly generated file with CIK codes
CIK_FILE = "geospatial_companies_with_cik.parquet"  # or use .csv if you prefer
REQUEST_DELAY = 0.5  # seconds between requests

headers = {
    "User-Agent": USER_AGENT
}

def load_tickers_with_cik():
    """Load tickers and CIK codes from your generated file"""
    logger.info(f"Loading tickers with CIK from {CIK_FILE}")
    
    # Try parquet first, fall back to CSV
    if Path(CIK_FILE).exists():
        df = pd.read_parquet(CIK_FILE)
    elif Path(CIK_FILE.replace('.parquet', '.csv')).exists():
        df = pd.read_csv(CIK_FILE.replace('.parquet', '.csv'))
    else:
        logger.error(f"File not found: {CIK_FILE}")
        return None
    
    # Filter only rows with valid CIK
    df = df[df['cik'].notna()].copy()
    
    logger.info(f"Loaded {len(df)} companies with CIK codes")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df

def fetch_filing_history(cik):
    """Fetch filing history for a given CIK from SEC's official API"""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            logger.warning(f"Submissions not found for CIK {cik}")
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
        return None

def parse_filings(submissions, ticker_filter=None, cik=None, form_types=None):
    """Parse filings from submissions JSON"""
    filings = []
    if not submissions or "filings" not in submissions:
        return filings

    recent = submissions["filings"].get("recent", {})
    count = len(recent.get("accessionNumber", []))
    
    for i in range(count):
        form = recent["form"][i]
        if form_types and form not in form_types:
            continue
        
        accession = recent["accessionNumber"][i]
        filing_date = recent.get("filingDate", [None]*count)[i]
        description = recent.get("primaryDocument", [form]*count)[i]
        
        filings.append({
            "ticker": ticker_filter,
            "cik": cik,
            "formType": form,
            "accessionNo": accession,
            "filedAt": filing_date,
            "description": description,
            "linkToFilingDetails": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}&dateb=&owner=exclude&count=100",
            "source": "SEC_SUBMISSIONS"
        })
    
    return filings

def etl_official_sec_api():
    """Main ETL function using official SEC API with your CIK file"""
    # Load your file with tickers and CIKs
    companies_df = load_tickers_with_cik()
    
    if companies_df is None:
        logger.error("Failed to load CIK file")
        return
    
    # Find the ticker column (handle different naming)
    ticker_col = None
    for col in ['ticker', 'Ticker', 'TICKER', 'symbol', 'Symbol']:
        if col in companies_df.columns:
            ticker_col = col
            break
    
    if ticker_col is None:
        logger.error(f"Could not find ticker column. Available: {companies_df.columns.tolist()}")
        return
    
    logger.info(f"Using ticker column: '{ticker_col}'")
    logger.info(f"Total companies with CIK: {len(companies_df)}")

    collected_filings = []
    form_types_of_interest = ["10-K", "10-Q", "8-K", "S-1", "13F-HR", "4"]

    # Fetch filings for each company
    for idx, row in companies_df.iterrows():
        ticker = row[ticker_col]
        cik = str(row['cik']).zfill(10)  # Ensure 10-digit CIK
        
        logger.info(f"[{idx+1}/{len(companies_df)}] Fetching filings for {ticker} (CIK: {cik})")
        submissions_json = fetch_filing_history(cik)
        
        if submissions_json:
            filings = parse_filings(
                submissions_json, 
                ticker_filter=ticker, 
                cik=cik,
                form_types=form_types_of_interest
            )
            collected_filings.extend(filings)
            logger.info(f"  → Collected {len(filings)} filings")
        
        time.sleep(REQUEST_DELAY)

    if not collected_filings:
        logger.warning("No filings collected")
        return

    # Create DataFrame
    df = pd.DataFrame(collected_filings)
    df['filedAt'] = pd.to_datetime(df['filedAt'], errors='coerce')
    
    # Filter to last 5 years
    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=5*365))
    df_5y = df[df['filedAt'] >= cutoff_date].copy()
    
    logger.info(f"\n📅 Filtering to last 5 years (from {cutoff_date.date()} onwards)")
    logger.info(f"  Before filter: {len(df):,} filings")
    logger.info(f"  After filter: {len(df_5y):,} filings")
    
    # Sort by filing date
    df_5y = df_5y.sort_values('filedAt', ascending=False)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as Parquet
    output_parquet = OUTPUT_DIR / f"sec_filings_last5y_{timestamp}.parquet"
    df_5y.to_parquet(output_parquet, index=False)
    logger.info(f"✓ Saved {len(df_5y)} filings to {output_parquet}")
    
    # Save as CSV
    output_csv = OUTPUT_DIR / f"sec_filings_last5y_{timestamp}.csv"
    df_5y.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved {len(df_5y)} filings to {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SEC FILINGS DATA SUMMARY - LAST 5 YEARS")
    print("="*60)
    
    print(f"\n📊 Dataset Overview (Last 5 Years):")
    print(f"  Total filings: {len(df_5y):,}")
    print(f"  Columns: {len(df_5y.columns)}")
    print(f"  Column names: {', '.join(df_5y.columns)}")
    
    print(f"\n🏢 Company Statistics:")
    print(f"  Unique tickers: {df_5y['ticker'].nunique()}")
    print(f"  Unique CIKs: {df_5y['cik'].nunique()}")
    
    print(f"\n📄 Form Type Distribution:")
    print(df_5y['formType'].value_counts())
    
    print(f"\n📅 Date Range (Last 5 Years):")
    print(f"  Earliest: {df_5y['filedAt'].min()}")
    print(f"  Latest: {df_5y['filedAt'].max()}")
    
    print(f"\n🔍 Sample Records:")
    print(df_5y[['ticker', 'cik', 'formType', 'filedAt', 'description']].head(10).to_string(index=False))
    
    logger.info(f"\n✓ Complete! Files saved in: {OUTPUT_DIR}")
    logger.info(f"  - CSV: {output_csv.name}")
    logger.info(f"  - Parquet: {output_parquet.name}")
    
    return df_5y

if __name__ == "__main__":
    etl_official_sec_api()
