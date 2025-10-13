
#!/usr/bin/env python3
"""
Geospatial Finance AI CLI Tool

A command-line interface for querying and analyzing geospatial finance data
using Claude AI for natural language insights and analysis.
"""

import click
import pandas as pd
import anthropic
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialFinanceAI:
    """AI-powered analysis of geospatial finance data"""

    def __init__(self, anthropic_api_key: str, data_dir: Path = Path('./geospatial_finance_data')):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.data_dir = Path(data_dir)
        self.latest_data = None

    def load_latest_data(self) -> Optional[pd.DataFrame]:
        """Load the most recent parquet file"""

        parquet_files = list(self.data_dir.glob('*.parquet'))

        if not parquet_files:
            logger.error("No parquet files found in data directory")
            return None

        # Get the most recent file
        latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_parquet(latest_file)
            self.latest_data = df
            logger.info(f"Loaded {len(df)} records from {latest_file.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def prepare_data_context(self, df: pd.DataFrame, query: str, max_rows: int = 50) -> str:
        """Prepare relevant data context for Claude"""

        # Filter data based on query keywords if needed
        query_lower = query.lower()
        relevant_keywords = ['insider', 'trading', 'filing', 'lobbying', '13f', 'acquisition', 'merger']

        filtered_df = df

        # Try to filter based on query context
        if any(keyword in query_lower for keyword in relevant_keywords):
            if 'insider' in query_lower or 'trading' in query_lower:
                filtered_df = df[df['category'].isin(['insider_trading', 'political_trading'])]
            elif 'filing' in query_lower:
                filtered_df = df[df['category'] == 'filing']
            elif 'lobbying' in query_lower:
                filtered_df = df[df['category'] == 'lobbying_activity']
            elif '13f' in query_lower:
                filtered_df = df[df['category'] == 'institutional_holding']

        # Limit rows for context
        if len(filtered_df) > max_rows:
            filtered_df = filtered_df.head(max_rows)

        # Create summary statistics
        summary_stats = {
            'total_records': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'date_range': {
                'start': df['filedAt'].min().strftime('%Y-%m-%d') if 'filedAt' in df.columns else 'N/A',
                'end': df['filedAt'].max().strftime('%Y-%m-%d') if 'filedAt' in df.columns else 'N/A'
            },
            'top_companies': df['companyName'].value_counts().head(10).to_dict() if 'companyName' in df.columns else {},
            'top_tickers': df['ticker'].value_counts().head(10).to_dict() if 'ticker' in df.columns else {}
        }

        # Format data for Claude
        context = f"""
GEOSPATIAL FINANCE DATA SUMMARY:
{json.dumps(summary_stats, indent=2, default=str)}

SAMPLE DATA RECORDS ({len(filtered_df)} records shown):
{filtered_df.to_string(max_rows=max_rows, max_cols=10)}
"""

        return context

    def query_with_claude(self, query: str, data_context: str) -> str:
        """Send query to Claude with data context"""

        system_prompt = """You are a financial analyst specializing in geospatial industry intelligence. 

        You have access to a comprehensive dataset containing:
        - SEC filings mentioning geospatial technologies
        - Insider trading activities 
        - 13F institutional holdings
        - Lobbying activities related to geospatial industry
        - Congressional trading disclosures
        - Company fundamental metrics

        Provide insightful analysis based on the data provided. Focus on:
        - Market trends and signals
        - Regulatory activity and policy implications  
        - Insider sentiment and institutional positioning
        - Political influence and lobbying patterns
        - Investment opportunities and risks

        Be specific and cite data points from the provided context when making claims.
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
                max_tokens=1000,
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
@click.option('--data-dir', default='./geospatial_finance_data', help='Directory containing parquet data files')
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
    df = ai.load_latest_data()

    if df is None:
        click.echo("❌ No data available")
        return

    click.echo("📊 GEOSPATIAL FINANCE DATA STATUS")
    click.echo("=" * 40)
    click.echo(f"Total records: {len(df):,}")
    click.echo(f"Date range: {df['filedAt'].min()} to {df['filedAt'].max()}")
    click.echo()

    click.echo("Categories:")
    for category, count in df['category'].value_counts().items():
        click.echo(f"  • {category}: {count:,}")

    click.echo()
    click.echo("Sources:")
    for source, count in df['source'].value_counts().items():
        click.echo(f"  • {source}: {count:,}")

    if 'ticker' in df.columns:
        click.echo()
        click.echo("Top companies by activity:")
        for ticker, count in df['ticker'].value_counts().head(5).items():
            click.echo(f"  • {ticker}: {count:,}")

@cli.command()
@click.argument('query')
@click.option('--max-context', default=50, help='Maximum rows to include in context')
@click.pass_context
def ask(ctx, query, max_context):
    """Ask Claude AI a question about the geospatial finance data"""

    ai = ctx.obj['ai']
    df = ai.load_latest_data()

    if df is None:
        click.echo("❌ No data available. Please run the ETL pipeline first.")
        return

    click.echo(f"🤔 Query: {query}")
    click.echo("🤖 Analyzing with Claude AI...")
    click.echo()

    # Prepare context and query Claude
    data_context = ai.prepare_data_context(df, query, max_context)
    response = ai.query_with_claude(query, data_context)

    click.echo("💡 CLAUDE'S ANALYSIS:")
    click.echo("=" * 50)
    click.echo(response)

@cli.command()
@click.option('--category', help='Filter by category')
@click.option('--ticker', help='Filter by ticker symbol')
@click.option('--days', default=7, help='Number of recent days to show')
@click.pass_context
def recent(ctx, category, ticker, days):
    """Show recent activity"""

    ai = ctx.obj['ai']
    df = ai.load_latest_data()

    if df is None:
        click.echo("❌ No data available")
        return

    # Filter data
    filtered_df = df.copy()

    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]

    if ticker:
        filtered_df = filtered_df[filtered_df['ticker'] == ticker]

    # Get recent data
    if 'filedAt' in filtered_df.columns:
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        filtered_df = filtered_df[filtered_df['filedAt'] >= cutoff_date]
        filtered_df = filtered_df.sort_values('filedAt', ascending=False)

    click.echo(f"📈 RECENT ACTIVITY (Last {days} days)")
    click.echo("=" * 50)

    if len(filtered_df) == 0:
        click.echo("No recent activity found")
        return

    for _, row in filtered_df.head(10).iterrows():
        click.echo(f"• {row.get('filedAt', 'N/A')} - {row.get('companyName', 'N/A')} ({row.get('ticker', 'N/A')})")
        click.echo(f"  {row.get('category', 'N/A')} - {row.get('source', 'N/A')}")
        if 'formType' in row:
            click.echo(f"  Form: {row['formType']}")
        click.echo()

@cli.command()
@click.argument('ticker')
@click.pass_context
def company(ctx, ticker):
    """Get detailed analysis for a specific company"""

    ai = ctx.obj['ai']
    df = ai.load_latest_data()

    if df is None:
        click.echo("❌ No data available")
        return

    # Filter for specific ticker
    company_data = df[df['ticker'].str.upper() == ticker.upper()]

    if len(company_data) == 0:
        click.echo(f"❌ No data found for {ticker}")
        return

    click.echo(f"🏢 COMPANY ANALYSIS: {ticker}")
    click.echo("=" * 50)

    # Basic stats
    company_name = company_data['companyName'].iloc[0] if 'companyName' in company_data.columns else ticker
    click.echo(f"Company: {company_name}")
    click.echo(f"Total records: {len(company_data)}")
    click.echo()

    # Activity by category
    click.echo("Activity breakdown:")
    for category, count in company_data['category'].value_counts().items():
        click.echo(f"  • {category}: {count}")

    # AI analysis
    click.echo()
    click.echo("🤖 AI ANALYSIS:")
    click.echo("-" * 30)

    context = ai.prepare_data_context(company_data, f"Analyze {ticker} activity", max_rows=20)
    response = ai.query_with_claude(f"Provide a detailed analysis of {ticker} based on recent activity", context)
    click.echo(response)

@cli.command()  
@click.pass_context
def trends(ctx):
    """Identify market trends and signals"""

    ai = ctx.obj['ai']
    df = ai.load_latest_data()

    if df is None:
        click.echo("❌ No data available")
        return

    click.echo("📈 MARKET TRENDS ANALYSIS")
    click.echo("=" * 50)

    # Prepare comprehensive context
    context = ai.prepare_data_context(df, "market trends analysis", max_rows=100)

    query = """
    Analyze the geospatial finance data to identify key market trends, signals, and insights. Focus on:

    1. Insider trading patterns - are executives buying or selling?
    2. Institutional positioning - what are 13F filers doing? 
    3. Regulatory activity - new filings, policy changes
    4. Lobbying trends - who's spending and on what issues?
    5. M&A activity and corporate developments
    6. Investment opportunities and risks

    Provide specific, actionable insights with data to support your conclusions.
    """

    response = ai.query_with_claude(query, context)
    click.echo(response)

if __name__ == '__main__':
    cli()
