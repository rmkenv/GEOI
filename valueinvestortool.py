#!/usr/bin/env python3
"""
Enhanced Value Investor CLI Tool for Geospatial Companies
- Uses geospatial_companies_with_cik.parquet from GitHub (includes CIK)
- Monitors SEC filings (Forms 4, 14A, 14C, S-1, 8-K)
- Google News monitoring for each stock
- Sentiment analysis for good/bad corporate conduct
- YTD, Quarterly, and 3-month projections
- Newsletter organized by industry vertical
- Conservative DCF valuation
- Supports free AI (HuggingFace) + Anthropic API backup
"""

import os
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, asdict
import urllib.request
import io
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ================================================================
# CACHES
# ================================================================
PRICE_CACHE = {}
FUNDAMENTAL_CACHE = {}
RECO_CACHE = {}
NEWS_CACHE = {}

# ================================================================
# DATA CLASSES
# ================================================================
@dataclass
class NewsArticle:
    ticker: str
    title: str
    source: str
    published_date: str
    url: str
    snippet: str
    sentiment: str  # Positive, Negative, Neutral
    conduct_flag: str  # Good, Bad, Neutral


@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    industry: str
    country: str
    index: str
    cik: Optional[str]
    current_price: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    market_cap: Optional[float]
    rsi: Optional[float]
    sma20: Optional[float]
    sma50: Optional[float]
    recommendation: str
    intrinsic_value: Optional[float]
    discount_vs_price: Optional[float]
    value_verdict: str
    ytd_return: Optional[float]
    quarterly_return: Optional[float]
    three_month_projection: str
    analysis: str
    news_summary: str
    conduct_assessment: str
    timestamp: str


@dataclass
class SECFiling:
    ticker: str
    form_type: str
    filing_date: str
    description: str
    url: str


# ================================================================
# PARQUET PARSING (Your GitHub Parquet with CIK)
# ================================================================
def load_geospatial_companies(parquet_url: str) -> pd.DataFrame:
    """Load geospatial companies from GitHub Parquet file"""
    logging.info(f"Loading Parquet from {parquet_url}...")
    
    try:
        # Download parquet file
        with urllib.request.urlopen(parquet_url) as resp:
            data = resp.read()
        
        # Read parquet
        df = pd.read_parquet(io.BytesIO(data))
        
        logging.info(f"Parquet columns detected: {list(df.columns)}")
        
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        
        # Map columns
        col_symbol = cols.get("symbol") or cols.get("ticker")
        col_company = cols.get("company name") or cols.get("company") or cols.get("name")
        col_industry = cols.get("industry") or cols.get("sector") or cols.get("vertical")
        col_country = cols.get("country") or cols.get("location")
        col_index = cols.get("index") or cols.get("exchange")
        # Correctly map CIK column from the new parquet file
        col_cik = cols.get("cik") 
        
        if not col_symbol:
            raise ValueError("Parquet must include a 'symbol' or 'ticker' column")
        
        # Build clean dataframe
        records = []
        for _, row in df.iterrows():
            sym = row.get(col_symbol)
            if pd.isna(sym):
                continue
            sym = str(sym).strip()
            if not sym:
                continue
            
            # Get CIK and clean it
            cik = None
            if col_cik:
                cik_val = row.get(col_cik)
                if not pd.isna(cik_val):
                    # Ensure CIK is treated as a string, remove leading zeros if it's a number
                    cik = str(int(float(cik_val))) if isinstance(cik_val, (int, float)) else str(cik_val).strip()
                    # Pad CIK to 10 digits with leading zeros for SEC API if it's not already
                    cik = cik.zfill(10)
            
            rec = {
                "Symbol": sym,
                "Company": str(row.get(col_company, "")).strip() if col_company else sym,
                "Industry": str(row.get(col_industry, "")).strip() if col_industry else "Unknown",
                "Country": str(row.get(col_country, "")).strip() if col_country else "Unknown",
                "Index": str(row.get(col_index, "")).strip() if col_index else "Unknown",
                "CIK": cik
            }
            records.append(rec)
        
        result = pd.DataFrame.from_records(records)
        result = result.drop_duplicates(subset=["Symbol"], keep="first")
        
        logging.info(f"Loaded {len(result)} unique companies")
        logging.info(f"Companies with CIK: {result['CIK'].notna().sum()}")
        logging.info(f"Industries: {result['Industry'].nunique()}")
        logging.info(f"Countries: {result['Country'].nunique()}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error loading Parquet: {e}")
        raise


# ================================================================
# AI CLIENT
# ================================================================
class AIClient:
    """Handles AI API calls with fallback support"""
    
    def __init__(self, use_anthropic: bool = False, api_key: Optional[str] = None):
        self.use_anthropic = use_anthropic
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        # Update to a known working Hugging Face model endpoint or provide a fallback
        # mistralai/Mistral-7B-Instruct-v0.2 might be behind a paywall or moved
        # Using a more generic or widely available model for free tier
        self.free_ai_model_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta" # A good alternative
        # self.free_ai_model_url = "https://api-inference.huggingface.co/models/google/gemma-7b-it" # Another option
        
    def analyze(self, prompt: str, max_retries: int = 2) -> str:
        """Send prompt to AI and get response"""
        if self.use_anthropic:
            return self._call_anthropic(prompt)
        else:
            return self._call_free_ai(prompt, max_retries)
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env variable.")
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    
    def _call_free_ai(self, prompt: str, max_retries: int = 2) -> str:
        """Call free AI API (using Hugging Face Inference API)"""
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.free_ai_model_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 503:
                    wait_time = 20 * (attempt + 1)
                    logging.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.warning(f"Free AI failed after {max_retries} attempts: {e}")
                    return "Analysis unavailable. Consider using --anthropic flag for better results."
                time.sleep(5)
        
        return "Analysis unavailable."


# ================================================================
# GOOGLE NEWS MONITOR
# ================================================================
class GoogleNewsMonitor:
    """Monitor Google News for stock mentions"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def get_news(self, ticker: str, company_name: str, days: int = 30) -> List[NewsArticle]:
        """Get recent news articles for a stock"""
        cache_key = f"{ticker}_{days}"
        if cache_key in NEWS_CACHE:
            return NEWS_CACHE[cache_key]
        
        articles = []
        
        try:
            # Use Google News RSS (no API key required)
            query = f"{company_name} OR {ticker}"
            url = f"https://news.google.com/rss/search?q={query}+when:30d&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logging.warning(f"Could not fetch news for {ticker}")
                return []
            
            # Parse RSS (simple XML parsing)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            for item in root.findall('.//item')[:10]:  # Limit to 10 articles
                title_elem = item.find('title')
                link_elem = item.find('link')
                pub_date_elem = item.find('pubDate')
                desc_elem = item.find('description')
                source_elem = item.find('source')
                
                if title_elem is None or link_elem is None:
                    continue
                
                title = title_elem.text
                url = link_elem.text
                pub_date = pub_date_elem.text if pub_date_elem is not None else ""
                snippet = desc_elem.text if desc_elem is not None else ""
                source = source_elem.text if source_elem is not None else "Google News"
                
                # Analyze sentiment and conduct
                sentiment, conduct = self._analyze_article(ticker, title, snippet)
                
                articles.append(NewsArticle(
                    ticker=ticker,
                    title=title,
                    source=source,
                    published_date=pub_date,
                    url=url,
                    snippet=snippet,
                    sentiment=sentiment,
                    conduct_flag=conduct
                ))
            
            NEWS_CACHE[cache_key] = articles
            logging.info(f"Found {len(articles)} news articles for {ticker}")
            
        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")
        
        return articles
    
    def _analyze_article(self, ticker: str, title: str, snippet: str) -> Tuple[str, str]:
        """Analyze article sentiment and corporate conduct"""
        text = f"{title}. {snippet}"
        
        # Simple keyword-based analysis (fast)
        positive_keywords = ['growth', 'profit', 'gain', 'success', 'innovation', 'award', 'partnership', 
                           'expansion', 'breakthrough', 'record', 'strong', 'beat', 'outperform']
        negative_keywords = ['loss', 'decline', 'lawsuit', 'scandal', 'fraud', 'investigation', 'layoff',
                           'bankruptcy', 'miss', 'weak', 'concern', 'warning', 'downgrade', 'cut']
        
        good_conduct = ['sustainability', 'ethical', 'charity', 'donation', 'community', 'diversity',
                       'transparency', 'responsible', 'green', 'renewable', 'social responsibility']
        bad_conduct = ['lawsuit', 'fraud', 'scandal', 'violation', 'fine', 'penalty', 'misconduct',
                      'corruption', 'discrimination', 'environmental damage', 'breach']
        
        text_lower = text.lower()
        
        # Sentiment
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Conduct
        good_count = sum(1 for kw in good_conduct if kw in text_lower)
        bad_count = sum(1 for kw in bad_conduct if kw in text_lower)
        
        if bad_count > 0:
            conduct = "Bad"
        elif good_count > 0:
            conduct = "Good"
        else:
            conduct = "Neutral"
        
        return sentiment, conduct
    
    def summarize_news(self, articles: List[NewsArticle]) -> Tuple[str, str]:
        """Generate news summary and conduct assessment"""
        if not articles:
            return "No recent news available.", "No conduct issues identified."
        
        # Count sentiments
        sentiments = [a.sentiment for a in articles]
        conducts = [a.conduct_flag for a in articles]
        
        pos = sentiments.count("Positive")
        neg = sentiments.count("Negative")
        neu = sentiments.count("Neutral")
        
        good = conducts.count("Good")
        bad = conducts.count("Bad")
        
        # News summary
        summary = f"Recent news coverage ({len(articles)} articles): "
        if pos > neg:
            summary += f"Predominantly positive ({pos} positive, {neg} negative). "
        elif neg > pos:
            summary += f"Predominantly negative ({neg} negative, {pos} positive). "
        else:
            summary += f"Mixed sentiment ({pos} positive, {neg} negative, {neu} neutral). "
        
        # Top headlines
        top_headlines = [a.title for a in articles[:3]]
        summary += f"Key headlines: {'; '.join(top_headlines)}"
        
        # Conduct assessment
        if bad > 0:
            bad_articles = [a for a in articles if a.conduct_flag == "Bad"]
            conduct_summary = f"⚠️ CONDUCT CONCERNS: {bad} article(s) flagged potential issues. "
            conduct_summary += f"Issues: {'; '.join([a.title for a in bad_articles[:2]])}"
        elif good > 0:
            conduct_summary = f"✓ POSITIVE CONDUCT: {good} article(s) highlight good corporate citizenship."
        else:
            conduct_summary = "No significant conduct issues identified in recent news."
        
        return summary, conduct_summary


# ================================================================
# SEC FILING MONITOR (Enhanced with CIK from Parquet)
# ================================================================
class SECFilingMonitor:
    """Monitor SEC EDGAR for filings"""
    
    BASE_URL = "https://data.sec.gov"
    
    def __init__(self):
        self.headers = {
            "User-Agent": "ValueInvestorTool/1.0 (investment-analysis@example.com)"
        }
        self.cik_cache = {}
    
    def get_recent_filings(self, ticker: str, cik: Optional[str] = None, days: int = 30) -> List[SECFiling]:
        """Get recent SEC filings for a ticker (uses CIK if provided)"""
        
        # Use provided CIK or fetch it
        if cik:
            cik_num = cik
            # CIK from parquet is already padded, no need to zfill here
            # logging.info(f"Using provided CIK for {ticker}: {cik_num}") 
        else:
            cik_num = self._get_cik(ticker)
            if not cik_num:
                logging.warning(f"No CIK found for {ticker}")
                return []
        
        url = f"{self.BASE_URL}/submissions/CIK{cik_num.zfill(10)}.json" # Ensure CIK is 10 digits for URL
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logging.warning(f"SEC API returned {response.status_code} for {ticker} (CIK: {cik_num})")
                return []
            
            data = response.json()
            filings = []
            
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            descriptions = recent.get("primaryDocument", [])
            
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            target_forms = ["4", "14A", "14C", "DEF 14A", "DEFM14A", "DEFR14A", "DEFA14A", 
                          "DEFC14A", "DEFN14A", "DEFR14C", "S-1", "S-1/A", "424B4", "8-K"]
            
            for i, (form, date, accession, desc) in enumerate(zip(forms, dates, accessions, descriptions)):
                if date < cutoff_date:
                    continue
                
                if form in target_forms:
                    # Create direct link to filing
                    accession_clean = accession.replace("-", "")
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_num)}/{accession_clean}/{accession}-index.htm"
                    
                    filings.append(SECFiling(
                        ticker=ticker,
                        form_type=form,
                        filing_date=date,
                        description=desc or form,
                        url=filing_url
                    ))
            
            if filings:
                logging.info(f"Found {len(filings)} SEC filings for {ticker}")
            
            return filings
            
        except Exception as e:
            logging.error(f"Error fetching SEC filings for {ticker}: {e}")
            return []
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for ticker (fallback if not in parquet)"""
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
        
        try:
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            for item in data.values():
                if item.get("ticker", "").upper() == ticker.upper():
                    cik = str(item.get("cik_str"))
                    self.cik_cache[ticker] = cik
                    return cik
        except Exception as e:
            logging.error(f"Error getting CIK for {ticker}: {e}")
        
        return None


# ================================================================
# FUNDAMENTALS & VALUATION
# ================================================================
def _safe_get(dct, key, default=None):
    try:
        return dct.get(key, default)
    except Exception:
        return default


def fetch_fundamentals(yf_symbol):
    """Fetch fundamental data with caching"""
    if yf_symbol in FUNDAMENTAL_CACHE:
        return FUNDAMENTAL_CACHE[yf_symbol]

    t = yf.Ticker(yf_symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    
    shares_out = _safe_get(info, "sharesOutstanding")
    long_name = _safe_get(info, "longName") or _safe_get(info, "shortName")
    industry = _safe_get(info, "industry")
    
    fcf_series = None
    try:
        cf = t.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            if "Free Cash Flow" in cf.index:
                fcf_series = cf.loc["Free Cash Flow"].dropna().astype(float)
            else:
                cfo = cf.loc["Total Cash From Operating Activities"].dropna().astype(float) if "Total Cash From Operating Activities" in cf.index else None
                capex = cf.loc["Capital Expenditures"].dropna().astype(float) if "Capital Expenditures" in cf.index else None
                if cfo is not None and capex is not None:
                    fcf_series = (cfo + capex).dropna()
    except Exception:
        pass

    result = {
        "info": info,
        "shares_out": shares_out,
        "long_name": long_name,
        "industry": industry,
        "fcf_series": fcf_series
    }
    FUNDAMENTAL_CACHE[yf_symbol] = result
    return result


def conservative_dcf_intrinsic(fcf_series, shares_out, base_growth=0.03, years=10, discount_rate=0.10, terminal_growth=0.02):
    """Conservative DCF valuation"""
    if fcf_series is None or len(fcf_series) == 0 or shares_out is None or shares_out <= 0:
        return None, None
    
    vals = pd.Series(fcf_series).dropna().astype(float).values
    start_fcf = float(np.mean(vals[:3])) if len(vals) >= 3 else float(vals[0])
    
    if start_fcf <= 0:
        return None, None
    
    fcf_list = [start_fcf * ((1 + base_growth) ** t) for t in range(1, years + 1)]
    discounts = [(1 + discount_rate) ** t for t in range(1, years + 1)]
    pv_fcf = sum(f / d for f, d in zip(fcf_list, discounts))
    
    tv = fcf_list[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth) if discount_rate > terminal_growth else 0.0
    pv_tv = tv / ((1 + discount_rate) ** years)
    
    equity_value = pv_fcf + pv_tv
    intrinsic_per_share = equity_value / shares_out
    
    return intrinsic_per_share, equity_value


def value_verdict(price, intrinsic, mos_threshold=0.30):
    """Determine value verdict with margin of safety"""
    if intrinsic is None or price is None or price <= 0:
        return "Too Hard to Value", np.nan
    
    discount = (intrinsic - price) / price
    
    if discount >= mos_threshold:
        return "Buy", discount
    if discount >= 0.0:
        return "Hold/Monitor", discount
    return "Sell/Avoid", discount


# ================================================================
# TECHNICAL INDICATORS
# ================================================================
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return None
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    
    return float(val) if not pd.isna(val) else None


def compute_indicators(hist):
    """Compute technical indicators"""
    hist = hist.copy()
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    rsi = calculate_rsi(hist["Close"], 14)
    
    close = float(hist["Close"].iloc[-1])
    sma20 = float(hist["SMA20"].iloc[-1]) if pd.notna(hist["SMA20"].iloc[-1]) else None
    sma50 = float(hist["SMA50"].iloc[-1]) if pd.notna(hist["SMA50"].iloc[-1]) else None
    
    return close, sma20, sma50, rsi


def get_recommendation(yf_symbol, hist_df=None, rsi=None, price=None, sma20=None, sma50=None):
    """Get recommendation with caching"""
    if yf_symbol in RECO_CACHE:
        return RECO_CACHE[yf_symbol]
    
    try:
        info = yf.Ticker(yf_symbol).info
        key = info.get("recommendationKey")
        if key:
            rec = key.replace("_", " ").title()
            RECO_CACHE[yf_symbol] = rec
            return rec
    except Exception:
        pass
    
    try:
        if hist_df is not None and (sma20 is None or sma50 is None or price is None or rsi is None):
            price, sma20, sma50, rsi = compute_indicators(hist_df)
    except Exception:
        pass
    
    rec = "Hold"
    if rsi and sma20 and sma50 and price:
        bull = (price > sma20 > sma50)
        bear = (price < sma20 < sma50)
        if rsi <= 30 and bull:
            rec = "Buy"
        elif rsi >= 70 and bear:
            rec = "Sell"
    
    RECO_CACHE[yf_symbol] = rec
    return rec


def calculate_period_return(ticker: str, period_days: int) -> Optional[float]:
    """Calculate return over a specific period"""
    try:
        t = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        hist = t.history(start=start_date, end=end_date)
        if hist.empty or len(hist) < 2:
            return None
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        
        return float((end_price - start_price) / start_price * 100)
    except Exception:
        return None


# ================================================================
# STOCK ANALYZER
# ================================================================
class StockAnalyzer:
    """Comprehensive stock analysis"""
    
    def __init__(self, ai_client: AIClient, news_monitor: GoogleNewsMonitor, 
                 discount_rate=0.10, base_growth=0.03, terminal_growth=0.02, mos_threshold=0.30):
        self.ai_client = ai_client
        self.news_monitor = news_monitor
        self.discount_rate = discount_rate
        self.base_growth = base_growth
        self.terminal_growth = terminal_growth
        self.mos_threshold = mos_threshold
    
    def analyze_stock(self, ticker: str, company_name: str, industry: str, country: str, 
                     index: str, cik: Optional[str] = None) -> Optional[StockAnalysis]:
        """Comprehensive stock analysis"""
        try:
            # Get price data
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                logging.warning(f"Insufficient data for {ticker}")
                return None
            
            # Technical indicators
            price, sma20, sma50, rsi = compute_indicators(hist)
            
            # Get info
            info = t.info or {}
            pe_ratio = info.get("trailingPE")
            pb_ratio = info.get("priceToBook")
            div_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None
            market_cap = info.get("marketCap")
            
            # Recommendation
            recommendation = get_recommendation(ticker, hist, rsi, price, sma20, sma50)
            
            # Fundamental valuation
            fundamentals = fetch_fundamentals(ticker)
            intrinsic, _ = conservative_dcf_intrinsic(
                fundamentals["fcf_series"],
                fundamentals["shares_out"],
                base_growth=self.base_growth,
                discount_rate=self.discount_rate,
                terminal_growth=self.terminal_growth
            )
            
            verdict, discount = value_verdict(price, intrinsic, self.mos_threshold)
            
            # Period returns
            ytd_return = calculate_period_return(ticker, 365)
            quarterly_return = calculate_period_return(ticker, 90)
            
            # News analysis
            news_articles = self.news_monitor.get_news(ticker, company_name)
            news_summary, conduct_assessment = self.news_monitor.summarize_news(news_articles)
            
            # 3-month projection using AI
            projection_prompt = f"""Based on the following data for {ticker} ({company_name}), provide a brief 2-3 sentence projection for the next 3 months:

Current Price: ${price:.2f}
YTD Return: {ytd_return:.1f}% if ytd_return is not None else 'N/A'
Quarterly Return: {quarterly_return:.1f}% if quarterly_return is not None else 'N/A'
Technical: RSI={rsi:.1f} if rsi is not None else 'N/A', Recommendation={recommendation}
Valuation: {verdict} (Intrinsic: ${intrinsic:.2f} if intrinsic is not None else 'N/A')
Recent News: {news_summary[:200]}

Provide a realistic 3-month outlook considering technical, fundamental, and news factors."""

            three_month_projection = self.ai_client.analyze(projection_prompt)
            
            # Overall AI Analysis
            analysis_prompt = f"""As a value investor, analyze {ticker} ({company_name}) in the {industry} sector:

**Valuation:**
- Price: ${price:.2f}
- P/E: {pe_ratio if pe_ratio else 'N/A'}
- P/B: {pb_ratio if pb_ratio else 'N/A'}
- Intrinsic Value: ${intrinsic:.2f} if intrinsic is not None else 'N/A'
- Verdict: {verdict}

**Performance:**
- YTD: {ytd_return:.1f}% if ytd_return is not None else 'N/A'
- Quarterly: {quarterly_return:.1f}% if quarterly_return is not None else 'N/A'

**News & Conduct:**
{news_summary[:300]}
{conduct_assessment}

Provide 3-4 sentences covering: (1) valuation assessment, (2) key risks/opportunities, (3) investment recommendation."""

            analysis_text = self.ai_client.analyze(analysis_prompt)
            
            return StockAnalysis(
                ticker=ticker,
                company_name=company_name,
                industry=industry,
                country=country,
                index=index,
                cik=cik,
                current_price=price,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                dividend_yield=div_yield,
                market_cap=market_cap,
                rsi=rsi,
                sma20=sma20,
                sma50=sma50,
                recommendation=recommendation,
                intrinsic_value=intrinsic,
                discount_vs_price=discount if pd.notna(discount) else None,
                value_verdict=verdict,
                ytd_return=ytd_return,
                quarterly_return=quarterly_return,
                three_month_projection=three_month_projection,
                analysis=analysis_text,
                news_summary=news_summary,
                conduct_assessment=conduct_assessment,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {e}")
            return None


# ================================================================
# NEWSLETTER GENERATOR (Organized by Vertical)
# ================================================================
class NewsletterGenerator:
    """Generate comprehensive monthly newsletter organized by industry vertical"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def generate_report(self, analyses: List[StockAnalysis], filings: List[SECFiling], 
                       output_file: str = "newsletter.md"):
        """Generate newsletter report organized by vertical"""
        
        # Group by industry
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# Geospatial Industry Investment Newsletter
## {datetime.now().strftime("%B %Y")}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        # Executive Summary
        if analyses:
            summary_prompt = f"""Based on analysis of {len(analyses)} geospatial companies across {len(by_industry)} industry verticals and {len(filings)} SEC filings, provide a 3-paragraph executive summary:

1. Overall market trends and opportunities in the geospatial sector
2. Key findings from news monitoring and corporate conduct
3. Top investment recommendations

Industries covered: {', '.join(by_industry.keys())}

Be concise and actionable for value investors."""

            summary = self.ai_client.analyze(summary_prompt)
            
            report += f"""## Executive Summary

{summary}

---

"""
        
        # Market Overview
        report += "## Market Overview\n\n"
        
        # Calculate aggregate metrics
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        
        report += f"""**Portfolio Statistics:**
- Total Companies Analyzed: {total_companies}
- Industry Verticals: {len(by_industry)}
- Average YTD Return: {avg_ytd:.2f}% if not np.isnan(avg_ytd) else 'N/A'
- Average Quarterly Return: {avg_quarterly:.2f}% if not np.isnan(avg_quarterly) else 'N/A'
- Buy Recommendations: {buy_count}
- Hold/Monitor: {hold_count}

---

"""
        
        # Analysis by Industry Vertical
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            
            report += f"""## {industry}

**Sector Overview:** {len(stocks)} companies analyzed

"""
            
            # Industry summary
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            ind_quarterly = np.nanmean([s.quarterly_return for s in stocks if s.quarterly_return is not None])
            
            report += f"""**Performance:**
- YTD Average: {ind_ytd:.2f}% if not np.isnan(ind_ytd) else 'N/A'
- Quarterly Average: {ind_quarterly:.2f}% if not np.isnan(ind_quarterly) else 'N/A'

**3-Month Sector Outlook:**

"""
            
            # Generate sector outlook
            sector_outlook_prompt = f"""Provide a 2-3 sentence outlook for the {industry} sector in the geospatial industry for the next 3 months based on:
- {len(stocks)} companies analyzed
- Average YTD return: {ind_ytd:.1f}% if not np.isnan(ind_ytd) else 'N/A'
- Average quarterly return: {ind_quarterly:.1f}% if not np.isnan(ind_quarterly) else 'N/A'

Be specific to geospatial technology trends."""
            
            sector_outlook = self.ai_client.analyze(sector_outlook_prompt)
            report += f"{sector_outlook}\n\n"
            
            report += "### Company Analysis\n\n"
            
            # Sort by discount to intrinsic value
            stocks_sorted = sorted(stocks, 
                                 key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                                 reverse=True)
            
            for stock in stocks_sorted:
                report += f"""#### {stock.ticker} - {stock.company_name}

**Location:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else 'N/A'}

**Valuation Metrics:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value (DCF): ${stock.intrinsic_value:.2f} if stock.intrinsic_value is not None else 'N/A'
- Discount to Price: {stock.discount_vs_price*100:.1f}% if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else 'N/A'
- **Value Verdict: {stock.value_verdict}**
- P/E Ratio: {stock.pe_ratio:.2f} if stock.pe_ratio is not None else 'N/A'
- P/B Ratio: {stock.pb_ratio:.2f} if stock.pb_ratio is not None else 'N/A'
- Dividend Yield: {stock.dividend_yield:.2f}% if stock.dividend_yield is not None else 'N/A'
- Market Cap: ${stock.market_cap:,.0f} if stock.market_cap is not None else 'N/A'

**Performance:**
- YTD Return: {stock.ytd_return:.2f}% if stock.ytd_return is not None else 'N/A'
- Quarterly Return: {stock.quarterly_return:.2f}% if stock.quarterly_return is not None else 'N/A'

**Technical Indicators:**
- RSI: {stock.rsi:.1f} if stock.rsi is not None else 'N/A'
- SMA20: ${stock.sma20:.2f} if stock.sma20 is not None else 'N/A'
- SMA50: ${stock.sma50:.2f} if stock.sma50 is not None else 'N/A'
- Recommendation: {stock.recommendation}

**News & Sentiment:**
{stock.news_summary}

**Corporate Conduct:**
{stock.conduct_assessment}

**3-Month Projection:**
{stock.three_month_projection}

**Investment Analysis:**
{stock.analysis}

---

"""
        
        # SEC Filings Section
        report += "\n## SEC Filings & Corporate Actions\n\n"
        
        if filings:
            filings_by_ticker = defaultdict(list)
            for filing in filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                
                # Find company name
                company_name = next((a.company_name for a in analyses if a.ticker == ticker), ticker)
                
                report += f"### {ticker} - {company_name}\n\n"
                
                for filing in sorted(ticker_filings, key=lambda x: x.filing_date, reverse=True):
                    form_desc = {
                        "4": "📊 Insider Trading",
                        "14A": "🗳️ Proxy Statement (Annual Meeting)",
                        "14C": "🗳️ Proxy Statement (Info)",
                        "DEF 14A": "🗳️ Definitive Proxy Statement",
                        "DEFM14A": "🗳️ Merger Proxy Statement",
                        "DEFR14A": "🗳️ Revised Proxy Statement",
                        "DEFA14A": "🗳️ Additional Proxy Materials",
                        "DEFC14A": "🗳️ Contested Proxy Statement",
                        "DEFN14A": "🗳️ Revised Definitive Proxy",
                        "DEFR14C": "🗳️ Revised Info Statement",
                        "S-1": "🚀 IPO Registration",
                        "S-1/A": "🚀 IPO Registration Amendment",
                        "424B4": "🚀 IPO Prospectus",
                        "8-K": "📢 Current Report"
                    }.get(filing.form_type, filing.form_type)
                    
                    report += f"- **{form_desc}** - {filing.filing_date}\n"
                    report += f"  [View Filing]({filing.url})\n\n"
        else:
            report += "*No significant filings in the monitoring period.*\n\n"
        
        # Methodology
        report += """---

## Methodology

**Valuation Approach:**
- Conservative DCF model using free cash flow
- 10-year projection with terminal value
- Margin of safety: 30% discount to intrinsic value
- Discount rate: 10% | Base growth: 3% | Terminal growth: 2%

**Technical Analysis:**
- RSI (14-period) for momentum
- SMA20/SMA50 for trend identification
- Combined with fundamental analysis for recommendations

**News Monitoring:**
- Google News RSS feeds for each company
- Sentiment analysis (Positive/Negative/Neutral)
- Corporate conduct assessment (Good/Bad/Neutral)
- 30-day lookback period

**SEC Monitoring:**
- Form 4: Insider trading activity
- Form 14A/14C: Proxy statements and shareholder votes
- Form S-1: IPO registrations
- Form 8-K: Material corporate events
- CIK numbers from geospatial_companies_with_cik.parquet

**Performance Metrics:**
- YTD: Year-to-date return
- Quarterly: 90-day return
- 3-Month Projection: AI-generated outlook based on technical, fundamental, and news factors

**Disclaimer:** This newsletter is for informational purposes only and does not constitute investment advice. The geospatial industry is subject to rapid technological change and regulatory developments. Always conduct your own due diligence and consult with a financial advisor before making investment decisions.

---

*Generated by Geospatial Value Investor CLI Tool*
*Data sources: Yahoo Finance, SEC EDGAR, Google News*
"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Newsletter generated: {output_file}")
        return output_file


# ================================================================
# MAIN CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Geospatial Industry Value Investor CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all companies from GitHub Parquet
  python valueinvestortool.py -o newsletter.md
  
  # Use Anthropic API for better analysis
  python valueinvestortool.py --anthropic
  
  # Custom parameters and longer lookback
  python valueinvestortool.py -d 60 --discount-rate 0.12 --mos-threshold 0.35
  
  # Limit to specific number of stocks
  python valueinvestortool.py --limit 50
  
  # Export to JSON/CSV
  python valueinvestortool.py --export-json analysis.json --export-csv analysis.csv
        """
    )
    
    # Updated default parquet URL
    parser.add_argument("--parquet-url", default="https://github.com/rmkenv/GEOI/raw/main/geospatial_companies_with_cik.parquet",
                       help="URL to geospatial companies Parquet file")
    parser.add_argument("--output", "-o", default="geospatial_newsletter.md", help="Output file for newsletter")
    parser.add_argument("--anthropic", action="store_true", help="Use Anthropic API instead of free AI")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days to look back for SEC filings and news")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to analyze (for testing)")
    
    # Valuation parameters
    parser.add_argument("--discount-rate", type=float, default=0.10, help="DCF discount rate (default: 0.10)")
    parser.add_argument("--base-growth", type=float, default=0.03, help="Base growth rate (default: 0.03)")
    parser.add_argument("--terminal-growth", type=float, default=0.02, help="Terminal growth rate (default: 0.02)")
    parser.add_argument("--mos-threshold", type=float, default=0.30, help="Margin of safety threshold (default: 0.30)")
    
    # Export options
    parser.add_argument("--export-json", help="Export analysis to JSON file")
    parser.add_argument("--export-csv", help="Export analysis to CSV file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🌍 GEOSPATIAL INDUSTRY VALUE INVESTOR CLI TOOL")
    print("   Enhanced with News Monitoring & Industry Vertical Analysis")
    print("=" * 80)
    print()
    
    # Load companies from GitHub Parquet
    companies_df = load_geospatial_companies(args.parquet_url)
    
    if args.limit:
        companies_df = companies_df.head(args.limit)
        logging.info(f"Limited to {args.limit} companies for analysis")
    
    print()
    
    # Initialize components
    ai_client = AIClient(use_anthropic=args.anthropic, api_key=args.api_key)
    news_monitor = GoogleNewsMonitor(ai_client)
    stock_analyzer = StockAnalyzer(
        ai_client,
        news_monitor,
        discount_rate=args.discount_rate,
        base_growth=args.base_growth,
        terminal_growth=args.terminal_growth,
        mos_threshold=args.mos_threshold
    )
    sec_monitor = SECFilingMonitor()
    newsletter_gen = NewsletterGenerator(ai_client)
    
    # Analyze stocks
    logging.info(f"Analyzing {len(companies_df)} geospatial companies...")
    analyses = []
    
    for i, row in companies_df.iterrows():
        ticker = row['Symbol']
        company = row['Company']
        industry = row['Industry']
        country = row['Country']
        index = row['Index']
        cik = row.get('CIK') # Get CIK from the loaded DataFrame
        
        logging.info(f"[{i+1}/{len(companies_df)}] Analyzing {ticker} - {company}...")
        
        analysis = stock_analyzer.analyze_stock(ticker, company, industry, country, index, cik)
        if analysis:
            analyses.append(analysis)
        
        # Rate limiting
        time.sleep(1)
    
    logging.info(f"Successfully analyzed {len(analyses)}/{len(companies_df)} companies")
    print()
    
    # Monitor SEC filings (using CIK from parquet)
    logging.info(f"Checking SEC filings (last {args.days} days)...")
    all_filings = []
    
    for i, row in companies_df.iterrows():
        ticker = row['Symbol']
        cik = row.get('CIK') # Get CIK from the loaded DataFrame
        
        logging.info(f"[{i+1}/{len(companies_df)}] Checking {ticker}...")
        
        filings = sec_monitor.get_recent_filings(ticker, cik=cik, days=args.days)
        all_filings.extend(filings)
        
        if filings:
            logging.info(f"  Found {len(filings)} filing(s)")
        
        time.sleep(0.5)  # Rate limiting
    
    logging.info(f"Found {len(all_filings)} total filings")
    print()
    
    # Generate newsletter
    logging.info("Generating newsletter organized by industry vertical...")
    newsletter_gen.generate_report(analyses, all_filings, args.output)
    
    # Export to JSON if requested
    if args.export_json:
        logging.info(f"Exporting to JSON: {args.export_json}")
        export_data = {
            "generated": datetime.now().isoformat(),
            "parameters": {
                "discount_rate": args.discount_rate,
                "base_growth": args.base_growth,
                "terminal_growth": args.terminal_growth,
                "mos_threshold": args.mos_threshold,
                "lookback_days": args.days
            },
            "summary": {
                "total_companies": len(companies_df),
                "analyzed": len(analyses),
                "industries": len(set(a.industry for a in analyses)),
                "sec_filings": len(all_filings)
            },
            "analyses": [asdict(a) for a in analyses],
            "filings": [asdict(f) for f in all_filings]
        }
        with open(args.export_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        logging.info(f"✓ Exported to {args.export_json}")
    
    # Export to CSV if requested
    if args.export_csv:
        logging.info(f"Exporting to CSV: {args.export_csv}")
        df = pd.DataFrame([asdict(a) for a in analyses])
        df.to_csv(args.export_csv, index=False)
        logging.info(f"✓ Exported to {args.export_csv}")
    
    print()
    print("=" * 80)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Newsletter: {args.output}")
    print(f"   Companies analyzed: {len(analyses)}/{len(companies_df)}")
    print(f"   Industry verticals: {len(set(a.industry for a in analyses))}")
    print(f"   SEC filings found: {len(all_filings)}")
    print(f"   News articles analyzed: {sum(len(NEWS_CACHE) for _ in [1])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
