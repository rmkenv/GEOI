
#!/usr/bin/env python3
"""
Geospatial Finance Dashboard

A Streamlit dashboard for visualizing geospatial finance intelligence data
and AI-powered insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Geospatial Finance Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DashboardData:
    """Data management for the dashboard"""

    def __init__(self, data_dir: Path = Path('./geospatial_finance_data')):
        self.data_dir = Path(data_dir)

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_data(_self) -> pd.DataFrame:
        """Load the most recent parquet file"""

        parquet_files = list(_self.data_dir.glob('*.parquet'))

        if not parquet_files:
            st.error("No data files found! Please run the ETL pipeline first.")
            return pd.DataFrame()

        # Get the most recent file
        latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_parquet(latest_file)

            # Ensure datetime columns are properly formatted
            datetime_cols = ['filedAt', 'transaction_date', 'disclosure_date', 'data_collection_timestamp']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

class AIInsights:
    """AI-powered insights using Claude"""

    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def generate_insights(self, df: pd.DataFrame, context: str) -> str:
        """Generate AI insights from the data"""

        if not self.client:
            return "⚠️ Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."

        # Prepare data summary
        summary = {
            'total_records': len(df),
            'categories': df['category'].value_counts().head(5).to_dict(),
            'recent_activity': df.head(10)[['ticker', 'companyName', 'category', 'filedAt']].to_dict('records') if len(df) > 0 else []
        }

        prompt = f"""
        Analyze this geospatial finance intelligence data and provide key insights:

        Context: {context}

        Data Summary:
        {json.dumps(summary, indent=2, default=str)}

        Provide 3-4 key insights focusing on market signals, trends, and actionable intelligence.
        """

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except Exception as e:
            return f"Error generating insights: {e}"

def create_metrics_cards(df: pd.DataFrame):
    """Create metric cards for the dashboard"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}",
            delta=f"{len(df[df['data_collection_timestamp'] >= datetime.now() - timedelta(days=1)]):,} today" if 'data_collection_timestamp' in df.columns else None
        )

    with col2:
        categories = df['category'].nunique() if 'category' in df.columns else 0
        st.metric(
            label="Data Categories",
            value=categories
        )

    with col3:
        companies = df['ticker'].nunique() if 'ticker' in df.columns else 0
        st.metric(
            label="Companies Tracked",
            value=companies
        )

    with col4:
        if 'filedAt' in df.columns:
            latest_date = df['filedAt'].max()
            days_ago = (datetime.now() - latest_date).days if pd.notna(latest_date) else 0
            st.metric(
                label="Latest Activity",
                value=f"{days_ago} days ago"
            )

def create_activity_timeline(df: pd.DataFrame):
    """Create activity timeline chart"""

    if 'filedAt' not in df.columns or len(df) == 0:
        st.warning("No timeline data available")
        return

    # Group by date and category
    df_timeline = df.copy()
    df_timeline['date'] = df_timeline['filedAt'].dt.date

    timeline_data = df_timeline.groupby(['date', 'category']).size().reset_index(name='count')

    fig = px.bar(
        timeline_data,
        x='date',
        y='count',
        color='category',
        title='Activity Timeline by Category',
        labels={'count': 'Number of Records', 'date': 'Date'}
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Activity Count",
        legend_title="Category"
    )

    st.plotly_chart(fig, use_container_width=True)

def create_company_activity_chart(df: pd.DataFrame):
    """Create company activity breakdown"""

    if 'ticker' not in df.columns or len(df) == 0:
        st.warning("No company data available")
        return

    # Top companies by activity
    company_activity = df['ticker'].value_counts().head(10)

    fig = px.bar(
        x=company_activity.values,
        y=company_activity.index,
        orientation='h',
        title='Top 10 Companies by Activity',
        labels={'x': 'Number of Records', 'y': 'Company Ticker'}
    )

    fig.update_layout(
        xaxis_title="Activity Count",
        yaxis_title="Company Ticker"
    )

    st.plotly_chart(fig, use_container_width=True)

def create_category_breakdown(df: pd.DataFrame):
    """Create category breakdown pie chart"""

    if 'category' not in df.columns or len(df) == 0:
        st.warning("No category data available")
        return

    category_counts = df['category'].value_counts()

    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Activity Breakdown by Category'
    )

    st.plotly_chart(fig, use_container_width=True)

def create_insider_trading_chart(df: pd.DataFrame):
    """Create insider trading analysis chart"""

    insider_data = df[df['category'] == 'insider_trading']

    if len(insider_data) == 0:
        st.info("No insider trading data available")
        return

    # Group by transaction type if available
    if 'transactionCode' in insider_data.columns:
        transaction_counts = insider_data['transactionCode'].value_counts()

        fig = px.bar(
            x=transaction_counts.index,
            y=transaction_counts.values,
            title='Insider Trading by Transaction Type',
            labels={'x': 'Transaction Code', 'y': 'Count'}
        )

        st.plotly_chart(fig, use_container_width=True)

    # Show recent insider trades
    st.subheader("Recent Insider Trading Activity")

    display_cols = ['filedAt', 'issuerTicker', 'reportingPerson', 'transactionCode', 'shares', 'totalValue']
    available_cols = [col for col in display_cols if col in insider_data.columns]

    if available_cols:
        recent_trades = insider_data[available_cols].head(10)
        st.dataframe(recent_trades, use_container_width=True)

def main():
    """Main dashboard function"""

    st.title("🌍 Geospatial Finance Intelligence Dashboard")
    st.markdown("Real-time monitoring of geospatial industry financial activity")

    # Initialize data loader and AI
    data_loader = DashboardData()
    ai = AIInsights()

    # Sidebar filters
    st.sidebar.header("Filters & Controls")

    # Load data
    df = data_loader.load_data()

    if df.empty:
        st.error("No data available. Please run the ETL pipeline first.")
        st.code("""
        # To generate data, run:
        python geospatial_finance_etl.py
        """)
        return

    # Date filter
    if 'filedAt' in df.columns:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(
                df['filedAt'].min().date(),
                df['filedAt'].max().date()
            ),
            min_value=df['filedAt'].min().date(),
            max_value=df['filedAt'].max().date()
        )

        if len(date_range) == 2:
            mask = (df['filedAt'].dt.date >= date_range[0]) & (df['filedAt'].dt.date <= date_range[1])
            df = df[mask]

    # Category filter
    if 'category' in df.columns:
        categories = st.sidebar.multiselect(
            "Categories",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        df = df[df['category'].isin(categories)]

    # Company filter
    if 'ticker' in df.columns:
        tickers = st.sidebar.multiselect(
            "Company Tickers",
            options=sorted(df['ticker'].dropna().unique()),
            default=None
        )
        if tickers:
            df = df[df['ticker'].isin(tickers)]

    # Main dashboard content
    st.header("📊 Overview")
    create_metrics_cards(df)

    # AI Insights section
    st.header("🤖 AI Insights")

    insight_type = st.selectbox(
        "Select Analysis Type",
        ["Market Overview", "Insider Trading Analysis", "Regulatory Activity", "Lobbying Trends"]
    )

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing data with Claude AI..."):
            insights = ai.generate_insights(df, insight_type)
            st.markdown(insights)

    # Charts section
    col1, col2 = st.columns(2)

    with col1:
        st.header("📈 Activity Timeline")
        create_activity_timeline(df)

    with col2:
        st.header("🏢 Company Activity")
        create_company_activity_chart(df)

    # Second row of charts
    col3, col4 = st.columns(2)

    with col3:
        st.header("📊 Category Breakdown")
        create_category_breakdown(df)

    with col4:
        st.header("💼 Insider Trading")
        create_insider_trading_chart(df)

    # Data table
    st.header("📋 Recent Activity")

    # Show recent records
    display_columns = ['filedAt', 'ticker', 'companyName', 'category', 'source', 'formType']
    available_columns = [col for col in display_columns if col in df.columns]

    if available_columns:
        recent_data = df[available_columns].sort_values('filedAt', ascending=False).head(20)
        st.dataframe(recent_data, use_container_width=True)

    # Data export
    st.header("💾 Data Export")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        if st.button("Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"geospatial_finance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col_export2:
        if st.button("Download JSON"):
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Click to Download",
                data=json_data,
                file_name=f"geospatial_finance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
