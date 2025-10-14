#!/usr/bin/env python3
"""
Geospatial Finance AI CLI Tool

A command-line interface for querying and analyzing geospatial finance data
using Claude AI for natural language insights and analysis.

Supports:
- SEC filings data
- IPO signals (S-1/F-1 prospectus analysis)
- Insider transactions (Forms 3, 4, 5)
"""

import click
import pandas as pd
import anthropic
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialFinanceAI:
    """AI-powered analysis of geospatial finance data"""

    def __init__(self, anthropic_api_key: str, data_dir: Path = Path('./geospatial_finance_data')):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.data_dir = Path(data_dir)
        self.filings_data = None
        self.ipo_data = None
        self.insider_data = None

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data files"""
        
        data = {}
        
        # Load SEC filings parquet
        parquet_files = list(self.data_dir.glob('sec_filings*.parquet'))
        if parquet_files:
            latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
            try:
                self.filings_data = pd.read_parquet(latest_file)
                data['filings'] = self.filings_data
                logger.info(f"Loaded {len(self.filings_data)} SEC filings from {latest_file.name}")
            except Exception as e:
                logger.error(f"Error loading filings: {e}")
        
        # Load IPO signals CSV
        ipo_file = self.data_dir / 'ipo_signals.csv'
        if ipo_file.exists():
            try:
                self.ipo_data = pd.read_csv(ipo_file)
                data['ipo'] = self.ipo_data
                logger.info(f"Loaded {len(self.ipo_data)} IPO signals")
            except Exception as e:
                logger.error(f"Error loading IPO data: {e}")
        
        # Load insider transactions CSV
        insider_file = self.data_dir / 'insider_transactions.csv'
        if insider_file.exists():
            try:
                self.insider_data = pd.read_csv(insider_file)
                # Convert transaction date to datetime
                if 'transactionDate' in self.insider_data.columns:
                    self.insider_data['transactionDate'] = pd.to_datetime(
                        self.insider_data['transactionDate'], errors='coerce'
                    )
                data['insider'] = self.insider_data
                logger.info(f"Loaded {len(self.insider_data)} insider transactions")
            except Exception as e:
                logger.error(f"Error loading insider data: {e}")
        
        if not data:
            logger.error("No data files found in data directory")
            return None
        
        return data

    def prepare_data_context(self, query: str, max_rows: int = 50) -> str:
        """Prepare relevant data context for Claude based on query"""
        
        query_lower = query.lower()
        context_parts = []
        
        # Determine which datasets are relevant
        include_filings = 'filing' in query_lower or 'sec' in query_lower
        include_ipo = 'ipo' in query_lower or 's-1' in query_lower or 'f-1' in query_lower or 'public' in query_lower
        include_insider = 'insider' in query_lower or 'trading' in query_lower or 'form 4' in query_lower or 'form 3' in query_lower
        
        # If no specific dataset mentioned, include all
        if not (include_filings or include_ipo or include_insider):
            include_filings = include_ipo = include_insider = True
        
        # SEC Filings Summary
        if include_filings and self.filings_data is not None:
            filings_summary = {
                'total_filings': len(self.filings_data),
                'form_types': self.filings_data['formType'].value_counts().head(10).to_dict(),
                'top_companies': self.filings_data['ticker'].value_counts().head(10).to_dict() if 'ticker' in self.filings_data.columns else {},
                'date_range': {
                    'start': str(self.filings_data['filedAt'].min()) if 'filedAt' in self.filings_data.columns else 'N/A',
                    'end': str(self.filings_data['filedAt'].max()) if 'filedAt' in self.filings_data.columns else 'N/A'
                }
            }
            context_parts.append(f"SEC FILINGS SUMMARY:\n{json.dumps(filings_summary, indent=2)}\n")
        
        # IPO Signals
        if include_ipo and self.ipo_data is not None and len(self.ipo_data) > 0:
            ipo_summary = {
                'total_ipo_filings': len(self.ipo_data),
                'companies_going_public': self.ipo_data['ticker'].unique().tolist(),
                'form_types': self.ipo_data['formType'].value_counts().to_dict(),
            }
            
            # Add pricing info if available
            if 'priceRangeLow' in self.ipo_data.columns:
                ipo_summary['companies_with_pricing'] = self.ipo_data[
                    self.ipo_data['priceRangeLow'].notna()
                ]['ticker'].tolist()
            
            context_parts.append(f"IPO SIGNALS SUMMARY:\n{json.dumps(ipo_summary, indent=2)}\n")
            
            # Add sample IPO records
            sample_ipo = self.ipo_data.head(max_rows).to_dict('records')
            context_parts.append(f"SAMPLE IPO RECORDS:\n{json.dumps(sample_ipo, indent=2, default=str)}\n")
        
        # Insider Transactions
        if include_insider and self.insider_data is not None and len(self.insider_data) > 0:
            # Filter for actual transactions (not just ownership reports)
            transactions = self.insider_data[
                self.insider_data['transactionCode'].notna() & 
                (self.insider_data['transactionCode'] != '')
            ].copy()
            
            insider_summary = {
                'total_transactions': len(transactions),
                'unique_companies': transactions['ticker'].nunique() if 'ticker' in transactions.columns else 0,
                'transaction_types': transactions['transactionCode'].value_counts().to_dict() if len(transactions) > 0 else {},
                'acquired_vs_disposed': transactions['acquiredDisposed'].value_counts().to_dict() if 'acquiredDisposed' in transactions.columns and len(transactions) > 0 else {},
            }
            
            # Calculate buy/sell sentiment
            if len(transactions) > 0 and 'acquiredDisposed' in transactions.columns:
                buys = len(transactions[transactions['acquiredDisposed'] == 'A'])
                sells = len(transactions[transactions['acquiredDisposed'] == 'D'])
                insider_summary['buy_sell_ratio'] = f"{buys} buys vs {sells} sells"
            
            # Top insider traders
            if 'ticker' in transactions.columns and len(transactions) > 0:
                insider_summary['most_active_companies'] = transactions['ticker'].value_counts().head(10).to_dict()
            
            context_parts.append(f"INSIDER TRADING SUMMARY:\n{json.dumps(insider_summary, indent=2)}\n")
            
            # Add recent transactions
            if 'transactionDate' in transactions.columns and len(transactions) > 0:
                recent_transactions = transactions.sort_values('transactionDate', ascending=False).head(max_rows)
                # Select key columns for context
                cols_to_show = ['ticker', 'transactionDate', 'transactionCode', 'shares', 'pricePerShare', 
                               'acquiredDisposed', 'reportingOwners', 'issuerName']
                cols_available = [col for col in cols_to_show if col in recent_transactions.columns]
                sample_insider = recent_transactions[cols_available].to_dict('records')
                context_parts.append(f"RECENT INSIDER TRANSACTIONS:\n{json.dumps(sample_insider, indent=2, default=str)}\n")
        
        return "\n".join(context_parts)

    def query_with_claude(self, query: str, data_context: str) -> str:
        """Send query to Claude with data context"""

        system_prompt = """You are a financial analyst specializing in geospatial industry intelligence. 

You have access to comprehensive datasets containing:
- SEC filings from geospatial companies
- IPO signals (S-1/F-1 filings) with offering details, pricing, underwriters, and intended exchanges
- Insider trading activities (Forms 3, 4, 5) showing executive and director transactions

For IPO analysis, focus on:
- Valuation signals (price ranges, offering amounts)
- Market timing and readiness
- Underwriter quality and syndicate
- Use of proceeds and growth plans

For insider trading analysis, focus on:
- Transaction codes: P=Purchase (bullish), S=Sale (bearish), A=Award, M=Option exercise
- Acquired (A) vs Disposed (D) sentiment
- Clustering of activity (multiple insiders buying/selling)
- Size and timing of transactions
- Officer titles and their significance

Provide specific, actionable insights citing data points. Identify patterns, anomalies, and investment signals.
"""

        user_prompt = f"""
Query: {query}

Data Context:
{data_context}

Please analyze this data and provide insights relevant to the query.
"""

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            return f"Error generating AI response: {e}"

@click.group()
@click.option('--data-dir', default='./geospatial_finance_data', help='Directory containing data files')
@click.option('--anthropic-key', envvar='ANTHROPIC_API_KEY', help='Anthropic API key')
@click.pass_context
def cli(ctx, data_dir, anthropic_key):
    """Geospatial Finance AI Analysis CLI"""

    if not anthropic_key:
        click.echo("Error: Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        ctx.exit(1)

    ctx.ensure_object(dict)
    ctx.obj['ai'] = GeospatialFinanceAI(anthropic_key, Path(data_dir))

@cli.command()
@click.pass_context  
def status(ctx):
    """Show data status and summary"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data:
        click.echo("❌ No data available")
        return

    click.echo("📊 GEOSPATIAL FINANCE DATA STATUS")
    click.echo("=" * 50)
    
    # SEC Filings
    if 'filings' in data:
        df = data['filings']
        click.echo(f"\n📄 SEC Filings: {len(df):,} records")
        if 'filedAt' in df.columns:
            click.echo(f"   Date range: {df['filedAt'].min()} to {df['filedAt'].max()}")
        if 'formType' in df.columns:
            click.echo("   Top form types:")
            for form, count in df['formType'].value_counts().head(5).items():
                click.echo(f"     • {form}: {count:,}")
    
    # IPO Signals
    if 'ipo' in data:
        ipo_df = data['ipo']
        click.echo(f"\n🚀 IPO Signals: {len(ipo_df):,} filings")
        click.echo(f"   Companies going public: {ipo_df['ticker'].nunique()}")
        if 'formType' in ipo_df.columns:
            click.echo("   Form types:")
            for form, count in ipo_df['formType'].value_counts().items():
                click.echo(f"     • {form}: {count}")
        
        # Show companies with pricing
        if 'priceRangeLow' in ipo_df.columns:
            priced = ipo_df[ipo_df['priceRangeLow'].notna()]
            if len(priced) > 0:
                click.echo(f"\n   Companies with pricing ({len(priced)}):")
                for _, row in priced.iterrows():
                    click.echo(f"     • {row['ticker']}: ${row['priceRangeLow']} - ${row['priceRangeHigh']}")
    
    # Insider Transactions
    if 'insider' in data:
        insider_df = data['insider']
        # Filter for actual transactions
        transactions = insider_df[
            insider_df['transactionCode'].notna() & 
            (insider_df['transactionCode'] != '')
        ]
        
        click.echo(f"\n💼 Insider Transactions: {len(transactions):,} transactions")
        click.echo(f"   Companies tracked: {transactions['ticker'].nunique()}")
        
        if 'acquiredDisposed' in transactions.columns:
            buys = len(transactions[transactions['acquiredDisposed'] == 'A'])
            sells = len(transactions[transactions['acquiredDisposed'] == 'D'])
            click.echo(f"   Buy/Sell: {buys:,} buys vs {sells:,} sells")
        
        if 'transactionCode' in transactions.columns:
            click.echo("   Transaction types:")
            for code, count in transactions['transactionCode'].value_counts().head(5).items():
                code_name = {
                    'P': 'Open Market Purchase',
                    'S': 'Open Market Sale',
                    'A': 'Grant/Award',
                    'M': 'Option Exercise',
                    'G': 'Gift',
                    'D': 'Disposition'
                }.get(code, code)
                click.echo(f"     • {code} ({code_name}): {count:,}")

@cli.command()
@click.argument('query')
@click.option('--max-context', default=50, help='Maximum rows to include in context')
@click.pass_context
def ask(ctx, query, max_context):
    """Ask Claude AI a question about the geospatial finance data"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data:
        click.echo("❌ No data available. Please run the ETL pipeline first.")
        return

    click.echo(f"🤔 Query: {query}")
    click.echo("🤖 Analyzing with Claude AI...")
    click.echo()

    # Prepare context and query Claude
    data_context = ai.prepare_data_context(query, max_context)
    response = ai.query_with_claude(query, data_context)

    click.echo("💡 CLAUDE'S ANALYSIS:")
    click.echo("=" * 50)
    click.echo(response)

@cli.command()
@click.pass_context
def ipos(ctx):
    """Show IPO pipeline and analysis"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data or 'ipo' not in data:
        click.echo("❌ No IPO data available")
        return

    ipo_df = data['ipo']
    
    click.echo("🚀 IPO PIPELINE ANALYSIS")
    click.echo("=" * 50)
    click.echo(f"Total IPO filings: {len(ipo_df)}")
    click.echo(f"Companies: {', '.join(ipo_df['ticker'].unique())}")
    click.echo()

    # Show details for each company
    for ticker in ipo_df['ticker'].unique():
        company_ipos = ipo_df[ipo_df['ticker'] == ticker]
        latest = company_ipos.iloc[-1]
        
        click.echo(f"📊 {ticker}")
        if 'priceRangeLow' in latest and pd.notna(latest['priceRangeLow']):
            click.echo(f"   Price Range: ${latest['priceRangeLow']} - ${latest['priceRangeHigh']}")
        if 'proposedMaxAggregateOfferingPrice' in latest and pd.notna(latest['proposedMaxAggregateOfferingPrice']):
            click.echo(f"   Offering Size: ${latest['proposedMaxAggregateOfferingPrice']}")
        if 'intendedExchange' in latest and pd.notna(latest['intendedExchange']):
            click.echo(f"   Exchange: {latest['intendedExchange']}")
        if 'underwriters' in latest and pd.notna(latest['underwriters']):
            click.echo(f"   Underwriters: {latest['underwriters']}")
        click.echo()

    # AI analysis
    click.echo("🤖 AI ANALYSIS:")
    click.echo("-" * 30)
    context = ai.prepare_data_context("IPO analysis", max_rows=100)
    response = ai.query_with_claude("Analyze the IPO pipeline for geospatial companies. What are the key signals and opportunities?", context)
    click.echo(response)

@cli.command()
@click.option('--ticker', help='Filter by ticker symbol')
@click.option('--days', default=30, help='Number of recent days to show')
@click.pass_context
def insider(ctx, ticker, days):
    """Show insider trading activity and sentiment"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data or 'insider' not in data:
        click.echo("❌ No insider trading data available")
        return

    insider_df = data['insider']
    
    # Filter for actual transactions
    transactions = insider_df[
        insider_df['transactionCode'].notna() & 
        (insider_df['transactionCode'] != '')
    ].copy()

    if ticker:
        transactions = transactions[transactions['ticker'].str.upper() == ticker.upper()]

    # Filter by date
    if 'transactionDate' in transactions.columns:
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        transactions = transactions[transactions['transactionDate'] >= cutoff_date]
        transactions = transactions.sort_values('transactionDate', ascending=False)

    click.echo(f"💼 INSIDER TRADING ACTIVITY (Last {days} days)")
    if ticker:
        click.echo(f"   Filtered for: {ticker}")
    click.echo("=" * 50)

    if len(transactions) == 0:
        click.echo("No recent insider transactions found")
        return

    # Summary stats
    buys = len(transactions[transactions['acquiredDisposed'] == 'A'])
    sells = len(transactions[transactions['acquiredDisposed'] == 'D'])
    click.echo(f"Total transactions: {len(transactions)}")
    click.echo(f"Buys: {buys} | Sells: {sells}")
    click.echo()

    # Show recent transactions
    click.echo("Recent Transactions:")
    for _, row in transactions.head(20).iterrows():
        date = row['transactionDate'].strftime('%Y-%m-%d') if pd.notna(row['transactionDate']) else 'N/A'
        ticker_sym = row.get('ticker', 'N/A')
        code = row.get('transactionCode', 'N/A')
        shares = row.get('shares', 'N/A')
        price = row.get('pricePerShare', 'N/A')
        action = '🟢 BUY' if row.get('acquiredDisposed') == 'A' else '🔴 SELL'
        owner = row.get('reportingOwners', 'N/A')
        
        click.echo(f"{action} {date} | {ticker_sym} | {owner}")
        click.echo(f"       {shares} shares @ ${price} | Code: {code}")
        click.echo()

    # AI analysis
    if not ticker:
        click.echo("🤖 AI SENTIMENT ANALYSIS:")
        click.echo("-" * 30)
        context = ai.prepare_data_context("insider trading sentiment", max_rows=100)
        response = ai.query_with_claude(
            f"Analyze insider trading sentiment over the last {days} days. What patterns do you see? Which companies show bullish or bearish signals?",
            context
        )
        click.echo(response)

@cli.command()
@click.argument('ticker')
@click.pass_context
def company(ctx, ticker):
    """Get detailed analysis for a specific company"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data:
        click.echo("❌ No data available")
        return

    click.echo(f"🏢 COMPANY ANALYSIS: {ticker.upper()}")
    click.echo("=" * 50)

    found_data = False

    # Check SEC filings
    if 'filings' in data:
        filings = data['filings'][data['filings']['ticker'].str.upper() == ticker.upper()]
        if len(filings) > 0:
            found_data = True
            click.echo(f"\n📄 SEC Filings: {len(filings)} records")
            if 'formType' in filings.columns:
                for form, count in filings['formType'].value_counts().head(5).items():
                    click.echo(f"   • {form}: {count}")

    # Check IPO data
    if 'ipo' in data:
        ipo = data['ipo'][data['ipo']['ticker'].str.upper() == ticker.upper()]
        if len(ipo) > 0:
            found_data = True
            click.echo(f"\n🚀 IPO Status: Filing detected")
            latest = ipo.iloc[-1]
            if 'priceRangeLow' in latest and pd.notna(latest['priceRangeLow']):
                click.echo(f"   Price Range: ${latest['priceRangeLow']} - ${latest['priceRangeHigh']}")
            if 'intendedExchange' in latest and pd.notna(latest['intendedExchange']):
                click.echo(f"   Exchange: {latest['intendedExchange']}")

    # Check insider trading
    if 'insider' in data:
        insider = data['insider'][data['insider']['ticker'].str.upper() == ticker.upper()]
        transactions = insider[insider['transactionCode'].notna() & (insider['transactionCode'] != '')]
        if len(transactions) > 0:
            found_data = True
            buys = len(transactions[transactions['acquiredDisposed'] == 'A'])
            sells = len(transactions[transactions['acquiredDisposed'] == 'D'])
            click.echo(f"\n💼 Insider Trading: {len(transactions)} transactions")
            click.echo(f"   Buys: {buys} | Sells: {sells}")
            click.echo(f"   Sentiment: {'🟢 BULLISH' if buys > sells else '🔴 BEARISH' if sells > buys else '⚪ NEUTRAL'}")

    if not found_data:
        click.echo(f"\n❌ No data found for {ticker}")
        return

    # AI analysis
    click.echo("\n🤖 AI ANALYSIS:")
    click.echo("-" * 30)
    
    # Create focused context for this company
    context_parts = []
    if 'filings' in data:
        filings = data['filings'][data['filings']['ticker'].str.upper() == ticker.upper()]
        if len(filings) > 0:
            context_parts.append(f"SEC Filings:\n{filings.head(20).to_json(orient='records', indent=2)}")
    
    if 'ipo' in data:
        ipo = data['ipo'][data['ipo']['ticker'].str.upper() == ticker.upper()]
        if len(ipo) > 0:
            context_parts.append(f"IPO Data:\n{ipo.to_json(orient='records', indent=2)}")
    
    if 'insider' in data:
        insider = data['insider'][data['insider']['ticker'].str.upper() == ticker.upper()]
        if len(insider) > 0:
            context_parts.append(f"Insider Transactions:\n{insider.head(50).to_json(orient='records', indent=2, default=str)}")
    
    context = "\n\n".join(context_parts)
    response = ai.query_with_claude(
        f"Provide a comprehensive analysis of {ticker}. What are the key signals, risks, and opportunities?",
        context
    )
    click.echo(response)

@cli.command()  
@click.pass_context
def trends(ctx):
    """Identify market trends and signals across all data"""

    ai = ctx.obj['ai']
    data = ai.load_all_data()

    if not data:
        click.echo("❌ No data available")
        return

    click.echo("📈 MARKET TRENDS ANALYSIS")
    click.echo("=" * 50)

    # Prepare comprehensive context
    context = ai.prepare_data_context("comprehensive market trends analysis", max_rows=100)

    query = """
Analyze the geospatial finance data to identify key market trends, signals, and insights. Focus on:

1. IPO Pipeline - Which companies are going public? What does this signal about market conditions?
2. Insider Trading Patterns - Are executives buying or selling? What's the overall sentiment?
3. Sector Activity - Which segments of geospatial tech are most active?
4. Investment Signals - What are the actionable opportunities and risks?
5. Market Timing - Is this a good time to invest in geospatial companies?

Provide specific, actionable insights with data to support your conclusions.
"""

    response = ai.query_with_claude(query, context)
    click.echo(response)

if __name__ == '__main__':
    cli()
