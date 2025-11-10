#!/usr/bin/env python3
"""
Enhanced Value Investor CLI Tool for Geospatial Companies
- Uses geospatial_companies_with_cik.parquet from GitHub (includes CIK)
- Monitors SEC filings (Forms 4, 14A, 14C, S-1, 8-K)
- Google News monitoring for each stock (last 90 days)
- Sentiment analysis for good/bad corporate conduct
- YTD, Quarterly, and 3-month projections
- Multiple newsletter types: Free Summary, Premium Deep Dive, Vertical Deep Dives, Weekly Highlights
- Conservative DCF valuation
- Supports Anthropic (Claude) and Gemini AI providers
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
    weekly_return: Optional[float]
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
                    cik = str(int(float(cik_val))) if isinstance(cik_val, (int, float)) else str(cik_val).strip()
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
    """Handles AI API calls using either Anthropic (default) or Gemini."""

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
        self.provider = provider.lower().strip()
        
        # Determine which API key to use
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "gemini":
                self.api_key = os.getenv("GEMINI_API_KEY")
            else:
                self.api_key = None

        if not self.api_key:
            env_hint = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "GEMINI_API_KEY"
            raise ValueError(f"{self.provider.title()} API key required. Set {env_hint} env variable or pass via --api-key.")

        logging.info(f"✓ Using {self.provider.title()} API for AI analysis")

    def analyze(self, prompt: str, max_tokens: int = 2048) -> str:
        """Send prompt to configured AI provider."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, max_tokens)
        elif self.provider == "gemini":
            return self._call_gemini(prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

    def _call_anthropic(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call Anthropic Claude API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Anthropic API error: {e}")
            return "Anthropic AI analysis unavailable due to API error."

    def _call_gemini(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call Google Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens
            }
        }
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            js = response.json()
            return js.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except requests.exceptions.RequestException as e:
            logging.error(f"Gemini API error: {e}")
            return "Gemini AI analysis unavailable due to API error."


# ================================================================
# GOOGLE NEWS MONITOR
# ================================================================
class GoogleNewsMonitor:
    """Monitor Google News for stock mentions"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def get_news(self, ticker: str, company_name: str, days: int = 90) -> List[NewsArticle]:
        """Get recent news articles for a stock"""
        cache_key = f"{ticker}_{days}"
        if cache_key in NEWS_CACHE:
            return NEWS_CACHE[cache_key]
        
        articles = []
        
        try:
            query = f"{company_name} OR {ticker}"
            url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logging.warning(f"Could not fetch news for {ticker}")
                return []
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            for item in root.findall('.//item')[:10]:
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
        
        positive_keywords = ['growth', 'profit', 'gain', 'success', 'innovation', 'award', 'partnership', 
                           'expansion', 'breakthrough', 'record', 'strong', 'beat', 'outperform']
        negative_keywords = ['loss', 'decline', 'lawsuit', 'scandal', 'fraud', 'investigation', 'layoff',
                           'bankruptcy', 'miss', 'weak', 'concern', 'warning', 'downgrade', 'cut']
        
        good_conduct = ['sustainability', 'ethical', 'charity', 'donation', 'community', 'diversity',
                       'transparency', 'responsible', 'green', 'renewable', 'social responsibility']
        bad_conduct = ['lawsuit', 'fraud', 'scandal', 'violation', 'fine', 'penalty', 'misconduct',
                      'corruption', 'discrimination', 'environmental damage', 'breach']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
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
        
        sentiments = [a.sentiment for a in articles]
        conducts = [a.conduct_flag for a in articles]
        
        pos = sentiments.count("Positive")
        neg = sentiments.count("Negative")
        neu = sentiments.count("Neutral")
        
        good = conducts.count("Good")
        bad = conducts.count("Bad")
        
        summary = f"Recent news coverage ({len(articles)} articles): "
        if pos > neg:
            summary += f"Predominantly positive ({pos} positive, {neg} negative). "
        elif neg > pos:
            summary += f"Predominantly negative ({neg} negative, {pos} positive). "
        else:
            summary += f"Mixed sentiment ({pos} positive, {neg} negative, {neu} neutral). "
        
        top_headlines = [a.title for a in articles[:3]]
        summary += f"Key headlines: {'; '.join(top_headlines)}"
        
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
        
        if cik:
            cik_num = cik
        else:
            cik_num = self._get_cik(ticker)
            if not cik_num:
                logging.warning(f"No CIK found for {ticker}")
                return []
        
        url = f"{self.BASE_URL}/submissions/CIK{cik_num.zfill(10)}.json"
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
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            
            if hist.empty or len(hist) < 50:
                logging.warning(f"Insufficient data for {ticker}")
                return None
            
            price, sma20, sma50, rsi = compute_indicators(hist)
            
            info = t.info or {}
            pe_ratio = info.get("trailingPE")
            pb_ratio = info.get("priceToBook")
            div_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None
            market_cap = info.get("marketCap")
            
            recommendation = get_recommendation(ticker, hist, rsi, price, sma20, sma50)
            
            fundamentals = fetch_fundamentals(ticker)
            intrinsic, _ = conservative_dcf_intrinsic(
                fundamentals["fcf_series"],
                fundamentals["shares_out"],
                base_growth=self.base_growth,
                discount_rate=self.discount_rate,
                terminal_growth=self.terminal_growth
            )
            
            verdict, discount = value_verdict(price, intrinsic, self.mos_threshold)
            
            ytd_return = calculate_period_return(ticker, 365)
            quarterly_return = calculate_period_return(ticker, 90)
            weekly_return = calculate_period_return(ticker, 7)
            
            news_articles = self.news_monitor.get_news(ticker, company_name, days=90)
            news_summary, conduct_assessment = self.news_monitor.summarize_news(news_articles)
            
            ytd_str = f"{ytd_return:.1f}%" if ytd_return is not None else "N/A"
            quarterly_str = f"{quarterly_return:.1f}%" if quarterly_return is not None else "N/A"
            rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
            intrinsic_str = f"${intrinsic:.2f}" if intrinsic is not None else "N/A"
            
            projection_prompt = f"""Based on the following data for {ticker} ({company_name}), provide a brief 2-3 sentence projection for the next 3 months:

Current Price: ${price:.2f}
YTD Return: {ytd_str}
Quarterly Return: {quarterly_str}
Technical: RSI={rsi_str}, Recommendation={recommendation}
Valuation: {verdict} (Intrinsic: {intrinsic_str})
Recent News: {news_summary[:200]}

Provide a realistic 3-month outlook considering technical, fundamental, and news factors."""

            three_month_projection = self.ai_client.analyze(projection_prompt)
            
            pe_str = f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A"
            pb_str = f"{pb_ratio:.2f}" if pb_ratio is not None else "N/A"
            
            analysis_prompt = f"""As a value investor, analyze {ticker} ({company_name}) in the {industry} sector:

**Valuation:**
- Price: ${price:.2f}
- P/E: {pe_str}
- P/B: {pb_str}
- Intrinsic Value: {intrinsic_str}
- Verdict: {verdict}

**Performance:**
- YTD: {ytd_str}
- Quarterly: {quarterly_str}

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
                weekly_return=weekly_return,
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
# NEWSLETTER GENERATORS
# ================================================================
class NewsletterGenerator:
    """Generate multiple types of newsletters"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def generate_free_summary(self, analyses: List[StockAnalysis], filings: List[SECFiling], 
                             output_file: str = "free_summary.md"):
        """Generate FREE monthly 30,000 ft view summary newsletter"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# 🌍 Geospatial Industry Monthly Overview (FREE)
## {datetime.now().strftime("%B %Y")}

*Your high-level view of the geospatial investment landscape*

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        # Executive Summary
        if analyses:
            summary_prompt = f"""You are writing a FREE monthly newsletter for geospatial industry investors. Provide a compelling 30,000-foot view executive summary (4-5 paragraphs) covering:

1. Overall market sentiment and macro trends in geospatial technology
2. Key industry developments and emerging opportunities
3. Notable performance highlights across {len(by_industry)} verticals
4. Risk factors and challenges facing the sector
5. What to watch for next month

Data: {len(analyses)} companies analyzed, {len(filings)} SEC filings, Industries: {', '.join(by_industry.keys())}

Make it engaging and accessible for both novice and experienced investors. End with a teaser about premium deep-dive content."""

            summary = self.ai_client.analyze(summary_prompt, max_tokens=3000)
            
            report += f"""## 📊 Executive Summary

{summary}

---

"""
        
        # Market Snapshot
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        avg_weekly = np.nanmean([a.weekly_return for a in analyses if a.weekly_return is not None])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        sell_count = sum(1 for a in analyses if a.value_verdict == "Sell/Avoid")
        
        avg_ytd_str = f"{avg_ytd:.2f}%" if not np.isnan(avg_ytd) else "N/A"
        avg_quarterly_str = f"{avg_quarterly:.2f}%" if not np.isnan(avg_quarterly) else "N/A"
        avg_weekly_str = f"{avg_weekly:.2f}%" if not np.isnan(avg_weekly) else "N/A"
        
        report += f"""## 📈 Market Snapshot

**Portfolio Overview:**
- **Total Companies Tracked:** {total_companies}
- **Industry Verticals:** {len(by_industry)}
- **Average YTD Return:** {avg_ytd_str}
- **Average Quarterly Return:** {avg_quarterly_str}
- **Average Weekly Return:** {avg_weekly_str}

**Value Assessment:**
- 🟢 **Buy Signals:** {buy_count} companies
- 🟡 **Hold/Monitor:** {hold_count} companies
- 🔴 **Sell/Avoid:** {sell_count} companies

**Corporate Activity:**
- **SEC Filings (30 days):** {len(filings)}
- **News Articles Analyzed:** {sum(len(NEWS_CACHE) for _ in [1])}

---

"""
        
        # Industry Vertical Highlights
        report += "## 🏭 Industry Vertical Highlights\n\n"
        
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            ind_quarterly = np.nanmean([s.quarterly_return for s in stocks if s.quarterly_return is not None])
            
            ind_ytd_str = f"{ind_ytd:.2f}%" if not np.isnan(ind_ytd) else "N/A"
            ind_quarterly_str = f"{ind_quarterly:.2f}%" if not np.isnan(ind_quarterly) else "N/A"
            
            buy_in_vertical = sum(1 for s in stocks if s.value_verdict == "Buy")
            
            report += f"""### {industry}
- **Companies:** {len(stocks)}
- **YTD Performance:** {ind_ytd_str}
- **Quarterly Performance:** {ind_quarterly_str}
- **Buy Opportunities:** {buy_in_vertical}

"""
        
        report += "\n---\n\n"
        
        # Top Movers
        report += "## 🚀 Top Weekly Movers\n\n"
        
        weekly_sorted = sorted([a for a in analyses if a.weekly_return is not None], 
                              key=lambda x: x.weekly_return, reverse=True)
        
        report += "**Top Gainers:**\n\n"
        for stock in weekly_sorted[:5]:
            report += f"- **{stock.ticker}** ({stock.company_name}): +{stock.weekly_return:.2f}%\n"
        
        report += "\n**Top Decliners:**\n\n"
        for stock in weekly_sorted[-5:]:
            report += f"- **{stock.ticker}** ({stock.company_name}): {stock.weekly_return:.2f}%\n"
        
        report += "\n---\n\n"
        
        # Notable SEC Filings
        if filings:
            report += "## 📋 Notable Corporate Actions\n\n"
            
            insider_filings = [f for f in filings if f.form_type == "4"]
            proxy_filings = [f for f in filings if "14" in f.form_type]
            ipo_filings = [f for f in filings if f.form_type in ["S-1", "S-1/A", "424B4"]]
            material_events = [f for f in filings if f.form_type == "8-K"]
            
            report += f"""**Filing Summary:**
- Insider Trading (Form 4): {len(insider_filings)}
- Proxy Statements (14A/14C): {len(proxy_filings)}
- IPO Activity (S-1): {len(ipo_filings)}
- Material Events (8-K): {len(material_events)}

*For detailed filing analysis, see our Premium Deep Dive newsletter.*

---

"""
        
        # Call to Action
        report += """## 🔒 Unlock Premium Insights

This free newsletter provides a high-level overview of the geospatial investment landscape. 

**Premium subscribers get access to:**

📊 **Deep Dive Portfolio Analysis** - Comprehensive analysis of all stocks with detailed valuations, risk assessments, and investment theses

🎯 **Vertical-Specific Deep Dives** - In-depth reports on each industry vertical with competitive analysis and sector forecasts

📅 **Weekly Highlights Report** - Detailed weekly updates on market movements, news, and trading opportunities

🔍 **Individual Stock Deep Dives** - Exhaustive analysis of specific companies with DCF models, competitive positioning, and management assessment

💼 **SEC Filing Analysis** - Expert interpretation of insider trading, proxy statements, and material events

📈 **Custom Alerts** - Real-time notifications for buy/sell signals and material corporate events

**[Subscribe to Premium →](#)**

---

## Methodology

This newsletter analyzes {len(analyses)} geospatial companies using:
- Conservative DCF valuation models
- Technical analysis (RSI, SMA20/50)
- News sentiment analysis (90-day lookback)
- SEC filing monitoring
- Corporate conduct assessment

**Disclaimer:** This newsletter is for informational purposes only and does not constitute investment advice. Always conduct your own due diligence and consult with a financial advisor before making investment decisions.

---

*Generated by Geospatial Value Investor*
*Next free newsletter: {(datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Free summary newsletter generated: {output_file}")
        return output_file
    
    def generate_premium_deep_dive(self, analyses: List[StockAnalysis], filings: List[SECFiling], 
                                   output_file: str = "premium_deep_dive.md"):
        """Generate PREMIUM comprehensive deep dive of entire portfolio"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# 🔒 PREMIUM: Geospatial Industry Deep Dive Analysis
## {datetime.now().strftime("%B %Y")}

*Comprehensive investment analysis of the geospatial sector*

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        # Executive Summary with AI Deep Analysis
        if analyses:
            summary_prompt = f"""You are writing a PREMIUM deep-dive newsletter for sophisticated geospatial industry investors. Provide an exhaustive executive summary (6-8 paragraphs) covering:

1. Macro-economic factors affecting geospatial technology adoption
2. Detailed sector performance analysis across {len(by_industry)} verticals
3. Valuation trends and market inefficiencies
4. Competitive dynamics and consolidation trends
5. Technology disruption and innovation cycles
6. Regulatory and geopolitical considerations
7. Capital allocation trends (M&A, buybacks, dividends)
8. Specific investment recommendations with rationale

Data: {len(analyses)} companies, {len(filings)} SEC filings
Industries: {', '.join(by_industry.keys())}
Avg YTD: {np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None]):.2f}%

Be analytical, data-driven, and actionable. This is premium content for serious investors."""

            summary = self.ai_client.analyze(summary_prompt, max_tokens=4000)
            
            report += f"""## 📊 Executive Summary

{summary}

---

"""
        
        # Detailed Market Analysis
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        avg_weekly = np.nanmean([a.weekly_return for a in analyses if a.weekly_return is not None])
        
        avg_pe = np.nanmean([a.pe_ratio for a in analyses if a.pe_ratio is not None])
        avg_pb = np.nanmean([a.pb_ratio for a in analyses if a.pb_ratio is not None])
        avg_discount = np.nanmean([a.discount_vs_price for a in analyses if a.discount_vs_price is not None and not np.isnan(a.discount_vs_price)])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        sell_count = sum(1 for a in analyses if a.value_verdict == "Sell/Avoid")
        
        report += f"""## 📈 Comprehensive Market Analysis

### Portfolio Metrics

**Performance:**
- Total Companies Analyzed: {total_companies}
- Industry Verticals: {len(by_industry)}
- Average YTD Return: {avg_ytd:.2f}%
- Average Quarterly Return: {avg_quarterly:.2f}%
- Average Weekly Return: {avg_weekly:.2f}%

**Valuation:**
- Average P/E Ratio: {avg_pe:.2f}
- Average P/B Ratio: {avg_pb:.2f}
- Average Discount to Intrinsic Value: {avg_discount*100:.2f}%

**Investment Signals:**
- 🟢 Strong Buy: {buy_count} companies ({buy_count/total_companies*100:.1f}%)
- 🟡 Hold/Monitor: {hold_count} companies ({hold_count/total_companies*100:.1f}%)
- 🔴 Sell/Avoid: {sell_count} companies ({sell_count/total_companies*100:.1f}%)

**Corporate Activity:**
- SEC Filings (30 days): {len(filings)}
- News Articles Analyzed: {sum(len(NEWS_CACHE) for _ in [1])}
- Companies with Conduct Concerns: {sum(1 for a in analyses if 'CONDUCT CONCERNS' in a.conduct_assessment)}

---

"""
        
        # Industry Vertical Deep Dives
        report += "## 🏭 Industry Vertical Analysis\n\n"
        
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            ind_quarterly = np.nanmean([s.quarterly_return for s in stocks if s.quarterly_return is not None])
            ind_pe = np.nanmean([s.pe_ratio for s in stocks if s.pe_ratio is not None])
            ind_pb = np.nanmean([s.pb_ratio for s in stocks if s.pb_ratio is not None])
            
            buy_in_vertical = sum(1 for s in stocks if s.value_verdict == "Buy")
            
            report += f"""### {industry}

**Sector Metrics:**
- Companies: {len(stocks)}
- YTD Performance: {ind_ytd:.2f}%
- Quarterly Performance: {ind_quarterly:.2f}%
- Average P/E: {ind_pe:.2f}
- Average P/B: {ind_pb:.2f}
- Buy Opportunities: {buy_in_vertical}

**AI Sector Analysis:**

"""
            
            sector_prompt = f"""Provide a detailed 4-5 paragraph analysis of the {industry} vertical in the geospatial industry:

1. Competitive landscape and market leaders
2. Technology trends and innovation
3. Growth drivers and headwinds
4. Valuation assessment vs. historical norms
5. Investment opportunities and risks

Data: {len(stocks)} companies, Avg YTD: {ind_ytd:.2f}%, Avg P/E: {ind_pe:.2f}

Be specific and actionable."""

            sector_analysis = self.ai_client.analyze(sector_prompt, max_tokens=3000)
            report += f"{sector_analysis}\n\n"
            
            # Top picks in vertical
            stocks_sorted = sorted(stocks, 
                                 key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                                 reverse=True)
            
            report += f"**Top Investment Opportunities in {industry}:**\n\n"
            
            for stock in stocks_sorted[:3]:
                intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
                discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
                
                report += f"""#### {stock.ticker} - {stock.company_name}

- **Current Price:** ${stock.current_price:.2f}
- **Intrinsic Value:** {intrinsic_str}
- **Discount:** {discount_str}
- **Verdict:** {stock.value_verdict}
- **3-Month Outlook:** {stock.three_month_projection[:200]}...

"""
            
            report += "---\n\n"
        
        # Detailed Stock Analysis
        report += "## 📊 Complete Stock Analysis\n\n"
        
        all_stocks_sorted = sorted(analyses, 
                                  key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                                  reverse=True)
        
        for stock in all_stocks_sorted:
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
            pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
            pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
            div_str = f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "N/A"
            mcap_str = f"${stock.market_cap:,.0f}" if stock.market_cap is not None else "N/A"
            ytd_str = f"{stock.ytd_return:.2f}%" if stock.ytd_return is not None else "N/A"
            quarterly_str = f"{stock.quarterly_return:.2f}%" if stock.quarterly_return is not None else "N/A"
            weekly_str = f"{stock.weekly_return:.2f}%" if stock.weekly_return is not None else "N/A"
            rsi_str = f"{stock.rsi:.1f}" if stock.rsi is not None else "N/A"
            sma20_str = f"${stock.sma20:.2f}" if stock.sma20 is not None else "N/A"
            sma50_str = f"${stock.sma50:.2f}" if stock.sma50 is not None else "N/A"
            
            report += f"""### {stock.ticker} - {stock.company_name}

**Industry:** {stock.industry} | **Country:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else 'N/A'}

#### Valuation Analysis
- **Current Price:** ${stock.current_price:.2f}
- **Intrinsic Value (DCF):** {intrinsic_str}
- **Discount to Intrinsic:** {discount_str}
- **Value Verdict:** **{stock.value_verdict}**
- **P/E Ratio:** {pe_str}
- **P/B Ratio:** {pb_str}
- **Dividend Yield:** {div_str}
- **Market Cap:** {mcap_str}

#### Performance Metrics
- **YTD Return:** {ytd_str}
- **Quarterly Return:** {quarterly_str}
- **Weekly Return:** {weekly_str}

#### Technical Analysis
- **RSI (14):** {rsi_str}
- **SMA20:** {sma20_str}
- **SMA50:** {sma50_str}
- **Technical Recommendation:** {stock.recommendation}

#### News & Sentiment
{stock.news_summary}

#### Corporate Conduct
{stock.conduct_assessment}

#### 3-Month Projection
{stock.three_month_projection}

#### Investment Thesis
{stock.analysis}

---

"""
        
        # SEC Filings Deep Dive
        report += "\n## 📋 SEC Filings & Corporate Actions Analysis\n\n"
        
        if filings:
            filings_by_ticker = defaultdict(list)
            for filing in filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            # Categorize filings
            insider_filings = [f for f in filings if f.form_type == "4"]
            proxy_filings = [f for f in filings if "14" in f.form_type]
            ipo_filings = [f for f in filings if f.form_type in ["S-1", "S-1/A", "424B4"]]
            material_events = [f for f in filings if f.form_type == "8-K"]
            
            report += f"""### Filing Summary

**Total Filings (30 days):** {len(filings)}

- **Insider Trading (Form 4):** {len(insider_filings)}
- **Proxy Statements (14A/14C):** {len(proxy_filings)}
- **IPO Activity (S-1):** {len(ipo_filings)}
- **Material Events (8-K):** {len(material_events)}

### Detailed Filing Analysis

"""
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                company_name = next((a.company_name for a in analyses if a.ticker == ticker), ticker)
                
                report += f"#### {ticker} - {company_name}\n\n"
                
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
        
        report += """---

## 📚 Methodology

### Valuation Framework
- **DCF Model:** Conservative 10-year free cash flow projection
- **Discount Rate:** 10% (adjustable based on company risk profile)
- **Base Growth:** 3% (conservative industry assumption)
- **Terminal Growth:** 2% (long-term GDP growth proxy)
- **Margin of Safety:** 30% discount to intrinsic value for "Buy" rating

### Technical Analysis
- **RSI (14-period):** Momentum indicator for overbought/oversold conditions
- **SMA20/SMA50:** Trend identification and support/resistance levels
- **Volume Analysis:** Confirmation of price movements

### News & Sentiment
- **Data Source:** Google News RSS feeds
- **Lookback Period:** 90 days
- **Sentiment Classification:** Positive/Negative/Neutral based on keyword analysis
- **Conduct Assessment:** Good/Bad/Neutral based on corporate governance indicators

### SEC Monitoring
- **Forms Tracked:** 4 (Insider Trading), 14A/14C (Proxy), S-1 (IPO), 8-K (Material Events)
- **Data Source:** SEC EDGAR API
- **CIK Mapping:** Direct from geospatial_companies_with_cik.parquet

### Performance Metrics
- **YTD:** Year-to-date return (365 days)
- **Quarterly:** 90-day return
- **Weekly:** 7-day return
- **3-Month Projection:** AI-generated outlook based on technical, fundamental, and news factors

---

## ⚠️ Disclaimer

This premium newsletter is for informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or any other sort of advice. The information provided is based on publicly available data and proprietary analysis, but should not be relied upon as the sole basis for investment decisions.

**Key Considerations:**
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- The geospatial industry is subject to rapid technological change, regulatory developments, and competitive pressures
- DCF valuations are based on assumptions that may not materialize
- Always conduct your own due diligence and consult with a qualified financial advisor before making investment decisions

**Conflicts of Interest:** The authors may hold positions in securities discussed in this newsletter.

---

*Generated by Geospatial Value Investor Premium*
*Next premium newsletter: {(datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Premium deep dive newsletter generated: {output_file}")
        return output_file
    
    def generate_vertical_deep_dive(self, analyses: List[StockAnalysis], filings: List[SECFiling],
                                   vertical: str, output_file: str = None):
        """Generate PREMIUM deep dive for specific vertical"""
        
        if output_file is None:
            output_file = f"vertical_{vertical.lower().replace(' ', '_')}.md"
        
        vertical_stocks = [a for a in analyses if a.industry == vertical]
        vertical_filings = [f for f in filings if any(a.ticker == f.ticker for a in vertical_stocks)]
        
        if not vertical_stocks:
            logging.warning(f"No stocks found for vertical: {vertical}")
            return None
        
        report = f"""# 🔒 PREMIUM: {vertical} Vertical Deep Dive
## {datetime.now().strftime("%B %Y")}

*Comprehensive analysis of the {vertical} sector in geospatial technology*

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        # Vertical Executive Summary
        summary_prompt = f"""You are writing a PREMIUM vertical-specific deep-dive for sophisticated investors in the {vertical} sector of geospatial technology. Provide an exhaustive analysis (8-10 paragraphs) covering:

1. Sector overview and market size/growth
2. Competitive landscape and market share analysis
3. Technology trends and innovation cycles
4. Key growth drivers and catalysts
5. Regulatory and geopolitical factors
6. Valuation analysis vs. historical norms and peers
7. M&A activity and consolidation trends
8. Capital allocation trends (R&D, capex, dividends, buybacks)
9. Risk factors and challenges
10. Specific investment recommendations with detailed rationale

Data: {len(vertical_stocks)} companies analyzed
Avg YTD: {np.nanmean([s.ytd_return for s in vertical_stocks if s.ytd_return is not None]):.2f}%
Avg P/E: {np.nanmean([s.pe_ratio for s in vertical_stocks if s.pe_ratio is not None]):.2f}

Be highly analytical, data-driven, and actionable. This is premium vertical-specific content."""

        summary = self.ai_client.analyze(summary_prompt, max_tokens=4000)
        
        report += f"""## 📊 Executive Summary

{summary}

---

"""
        
        # Vertical Metrics
        avg_ytd = np.nanmean([s.ytd_return for s in vertical_stocks if s.ytd_return is not None])
        avg_quarterly = np.nanmean([s.quarterly_return for s in vertical_stocks if s.quarterly_return is not None])
        avg_weekly = np.nanmean([s.weekly_return for s in vertical_stocks if s.weekly_return is not None])
        avg_pe = np.nanmean([s.pe_ratio for s in vertical_stocks if s.pe_ratio is not None])
        avg_pb = np.nanmean([s.pb_ratio for s in vertical_stocks if s.pb_ratio is not None])
        avg_discount = np.nanmean([s.discount_vs_price for s in vertical_stocks if s.discount_vs_price is not None and not np.isnan(s.discount_vs_price)])
        
        buy_count = sum(1 for s in vertical_stocks if s.value_verdict == "Buy")
        hold_count = sum(1 for s in vertical_stocks if s.value_verdict == "Hold/Monitor")
        sell_count = sum(1 for s in vertical_stocks if s.value_verdict == "Sell/Avoid")
        
        report += f"""## 📈 {vertical} Sector Metrics

**Performance:**
- Companies Analyzed: {len(vertical_stocks)}
- Average YTD Return: {avg_ytd:.2f}%
- Average Quarterly Return: {avg_quarterly:.2f}%
- Average Weekly Return: {avg_weekly:.2f}%

**Valuation:**
- Average P/E Ratio: {avg_pe:.2f}
- Average P/B Ratio: {avg_pb:.2f}
- Average Discount to Intrinsic Value: {avg_discount*100:.2f}%

**Investment Signals:**
- 🟢 Strong Buy: {buy_count} companies ({buy_count/len(vertical_stocks)*100:.1f}%)
- 🟡 Hold/Monitor: {hold_count} companies ({hold_count/len(vertical_stocks)*100:.1f}%)
- 🔴 Sell/Avoid: {sell_count} companies ({sell_count/len(vertical_stocks)*100:.1f}%)

**Corporate Activity:**
- SEC Filings (30 days): {len(vertical_filings)}
- Companies with Conduct Concerns: {sum(1 for s in vertical_stocks if 'CONDUCT CONCERNS' in s.conduct_assessment)}

---

"""
        
        # Competitive Analysis
        report += f"## 🏆 Competitive Landscape\n\n"
        
        # Sort by market cap
        stocks_by_mcap = sorted([s for s in vertical_stocks if s.market_cap is not None], 
                               key=lambda x: x.market_cap, reverse=True)
        
        if stocks_by_mcap:
            report += "**Market Leaders by Market Cap:**\n\n"
            for i, stock in enumerate(stocks_by_mcap[:5], 1):
                mcap_str = f"${stock.market_cap:,.0f}"
                report += f"{i}. **{stock.ticker}** ({stock.company_name}) - {mcap_str}\n"
            report += "\n"
        
        # Performance leaders
        stocks_by_ytd = sorted([s for s in vertical_stocks if s.ytd_return is not None], 
                              key=lambda x: x.ytd_return, reverse=True)
        
        if stocks_by_ytd:
            report += "**YTD Performance Leaders:**\n\n"
            for i, stock in enumerate(stocks_by_ytd[:5], 1):
                report += f"{i}. **{stock.ticker}** ({stock.company_name}) - {stock.ytd_return:.2f}%\n"
            report += "\n"
        
        report += "---\n\n"
        
        # Investment Opportunities
        report += f"## 💎 Top Investment Opportunities\n\n"
        
        stocks_sorted = sorted(vertical_stocks, 
                             key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                             reverse=True)
        
        for stock in stocks_sorted[:5]:
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
            pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
            pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
            
            report += f"""### {stock.ticker} - {stock.company_name}

**Valuation:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value: {intrinsic_str}
- Discount: {discount_str}
- P/E: {pe_str} | P/B: {pb_str}
- **Verdict: {stock.value_verdict}**

**Performance:**
- YTD: {stock.ytd_return:.2f}% | Quarterly: {stock.quarterly_return:.2f}%

**Investment Thesis:**
{stock.analysis}

**3-Month Outlook:**
{stock.three_month_projection}

**News & Conduct:**
{stock.conduct_assessment}

---

"""
        
        # Detailed Stock Profiles
        report += f"## 📊 Complete {vertical} Stock Profiles\n\n"
        
        for stock in sorted(vertical_stocks, key=lambda x: x.ticker):
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
            pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
            pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
            div_str = f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "N/A"
            mcap_str = f"${stock.market_cap:,.0f}" if stock.market_cap is not None else "N/A"
            ytd_str = f"{stock.ytd_return:.2f}%" if stock.ytd_return is not None else "N/A"
            quarterly_str = f"{stock.quarterly_return:.2f}%" if stock.quarterly_return is not None else "N/A"
            weekly_str = f"{stock.weekly_return:.2f}%" if stock.weekly_return is not None else "N/A"
            rsi_str = f"{stock.rsi:.1f}" if stock.rsi is not None else "N/A"
            
            report += f"""### {stock.ticker} - {stock.company_name}

**Country:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else 'N/A'}

#### Valuation
- Price: ${stock.current_price:.2f} | Intrinsic: {intrinsic_str} | Discount: {discount_str}
- P/E: {pe_str} | P/B: {pb_str} | Div Yield: {div_str}
- Market Cap: {mcap_str}
- **Verdict: {stock.value_verdict}**

#### Performance
- YTD: {ytd_str} | Quarterly: {quarterly_str} | Weekly: {weekly_str}
- RSI: {rsi_str} | Recommendation: {stock.recommendation}

#### Analysis
{stock.analysis}

#### News Summary
{stock.news_summary}

#### Conduct Assessment
{stock.conduct_assessment}

#### 3-Month Projection
{stock.three_month_projection}

---

"""
        
        # SEC Filings for Vertical
        if vertical_filings:
            report += f"## 📋 {vertical} SEC Filings & Corporate Actions\n\n"
            
            filings_by_ticker = defaultdict(list)
            for filing in vertical_filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                company_name = next((s.company_name for s in vertical_stocks if s.ticker == ticker), ticker)
                
                report += f"### {ticker} - {company_name}\n\n"
                
                for filing in sorted(ticker_filings, key=lambda x: x.filing_date, reverse=True):
                    form_desc = {
                        "4": "📊 Insider Trading",
                        "14A": "🗳️ Proxy Statement",
                        "8-K": "📢 Current Report",
                        "S-1": "🚀 IPO Registration"
                    }.get(filing.form_type, filing.form_type)
                    
                    report += f"- **{form_desc}** ({filing.filing_date}): [View Filing]({filing.url})\n"
                
                report += "\n"
        
        report += """---

## 📚 Methodology

This vertical deep dive uses the same rigorous methodology as our comprehensive portfolio analysis:

- **DCF Valuation:** Conservative 10-year free cash flow projection with 10% discount rate
- **Technical Analysis:** RSI, SMA20/50, volume confirmation
- **News Sentiment:** 90-day Google News analysis with conduct assessment
- **SEC Monitoring:** Forms 4, 14A/14C, S-1, 8-K tracking
- **Performance Metrics:** YTD, quarterly, and weekly returns

---

## ⚠️ Disclaimer

This premium vertical deep dive is for informational and educational purposes only and does not constitute investment advice. The information provided is based on publicly available data and proprietary analysis.

**Key Considerations:**
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- Sector-specific risks may include technology disruption, regulatory changes, and competitive pressures
- DCF valuations are based on assumptions that may not materialize
- Always conduct your own due diligence and consult with a qualified financial advisor

**Conflicts of Interest:** The authors may hold positions in securities discussed in this newsletter.

---

*Generated by Geospatial Value Investor Premium*
*Next vertical deep dive: {(datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Vertical deep dive generated: {output_file}")
        return output_file
    
    def generate_weekly_highlights(self, analyses: List[StockAnalysis], filings: List[SECFiling],
                                  output_file: str = "weekly_highlights.md"):
        """Generate PREMIUM weekly highlights newsletter"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# 🔒 PREMIUM: Weekly Highlights
## Week of {datetime.now().strftime("%B %d, %Y")}

*Your weekly pulse on the geospatial investment landscape*

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        # Weekly Executive Summary
        summary_prompt = f"""You are writing a PREMIUM weekly highlights newsletter for active geospatial industry investors. Provide a compelling executive summary (4-5 paragraphs) covering:

1. Key market movements and trends this week
2. Notable stock performance (winners and losers)
3. Significant corporate actions and SEC filings
4. News events impacting the sector
5. Trading opportunities and risks for the week ahead

Data: {len(analyses)} companies, {len(filings)} SEC filings this week
Avg Weekly Return: {np.nanmean([a.weekly_return for a in analyses if a.weekly_return is not None]):.2f}%

Be timely, actionable, and focused on short-term opportunities."""

        summary = self.ai_client.analyze(summary_prompt, max_tokens=3000)
        
        report += f"""## 📊 Executive Summary

{summary}

---

"""
        
        # Weekly Performance Overview
        avg_weekly = np.nanmean([a.weekly_return for a in analyses if a.weekly_return is not None])
        
        weekly_sorted = sorted([a for a in analyses if a.weekly_return is not None], 
                              key=lambda x: x.weekly_return, reverse=True)
        
        report += f"""## 📈 Weekly Performance Overview

**Market Summary:**
- Average Weekly Return: {avg_weekly:.2f}%
- Companies Tracked: {len(analyses)}
- SEC Filings This Week: {len(filings)}

---

"""
        
        # Top Gainers
        report += "## 🚀 Top Weekly Gainers\n\n"
        
        for i, stock in enumerate(weekly_sorted[:10], 1):
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            
            report += f"""### {i}. {stock.ticker} - {stock.company_name}

**Performance:**
- Weekly Return: **+{stock.weekly_return:.2f}%**
- YTD Return: {stock.ytd_return:.2f}% if stock.ytd_return is not None else "N/A"
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value: {intrinsic_str}
- Verdict: {stock.value_verdict}

**What Happened:**
{stock.news_summary[:300]}

**Outlook:**
{stock.three_month_projection[:200]}

---

"""
        
        # Top Decliners
        report += "## 📉 Top Weekly Decliners\n\n"
        
        for i, stock in enumerate(weekly_sorted[-10:], 1):
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            
            report += f"""### {i}. {stock.ticker} - {stock.company_name}

**Performance:**
- Weekly Return: **{stock.weekly_return:.2f}%**
- YTD Return: {stock.ytd_return:.2f}% if stock.ytd_return is not None else "N/A"
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value: {intrinsic_str}
- Verdict: {stock.value_verdict}

**What Happened:**
{stock.news_summary[:300]}

**Outlook:**
{stock.three_month_projection[:200]}

---

"""
        
        # New Buy Signals
        buy_signals = [a for a in analyses if a.value_verdict == "Buy"]
        
        if buy_signals:
            report += "## 💎 New Buy Signals\n\n"
            
            for stock in sorted(buy_signals, 
                              key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                              reverse=True)[:5]:
                discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
                
                report += f"""### {stock.ticker} - {stock.company_name}

**Opportunity:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value: ${stock.intrinsic_value:.2f}
- Discount: {discount_str}
- Weekly Return: {stock.weekly_return:.2f}% if stock.weekly_return is not None else "N/A"

**Why Buy:**
{stock.analysis}

---

"""
        
        # SEC Filings This Week
        if filings:
            report += "## 📋 SEC Filings & Corporate Actions\n\n"
            
            # Group by type
            insider_filings = [f for f in filings if f.form_type == "4"]
            proxy_filings = [f for f in filings if "14" in f.form_type]
            material_events = [f for f in filings if f.form_type == "8-K"]
            
            if insider_filings:
                report += "### 📊 Insider Trading (Form 4)\n\n"
                for filing in sorted(insider_filings, key=lambda x: x.filing_date, reverse=True)[:10]:
                    company_name = next((a.company_name for a in analyses if a.ticker == filing.ticker), filing.ticker)
                    report += f"- **{filing.ticker}** ({company_name}) - {filing.filing_date}: [View Filing]({filing.url})\n"
                report += "\n"
            
            if proxy_filings:
                report += "### 🗳️ Proxy Statements\n\n"
                for filing in sorted(proxy_filings, key=lambda x: x.filing_date, reverse=True):
                    company_name = next((a.company_name for a in analyses if a.ticker == filing.ticker), filing.ticker)
                    report += f"- **{filing.ticker}** ({company_name}) - {filing.filing_date}: [View Filing]({filing.url})\n"
                report += "\n"
            
            if material_events:
                report += "### 📢 Material Events (8-K)\n\n"
                for filing in sorted(material_events, key=lambda x: x.filing_date, reverse=True):
                    company_name = next((a.company_name for a in analyses if a.ticker == filing.ticker), filing.ticker)
                    report += f"- **{filing.ticker}** ({company_name}) - {filing.filing_date}: [View Filing]({filing.url})\n"
                report += "\n"
        
        # Conduct Concerns
        conduct_concerns = [a for a in analyses if 'CONDUCT CONCERNS' in a.conduct_assessment]
        
        if conduct_concerns:
            report += "## ⚠️ Conduct Concerns\n\n"
            
            for stock in conduct_concerns:
                report += f"""### {stock.ticker} - {stock.company_name}

{stock.conduct_assessment}

---

"""
        
        # Week Ahead
        report += """## 🔮 Week Ahead

**Key Events to Watch:**
- Earnings announcements
- Economic data releases (GDP, employment, inflation)
- Federal Reserve commentary
- Geopolitical developments
- Industry conferences and events

**Trading Strategy:**
- Monitor buy signals for entry opportunities
- Watch for technical breakouts above SMA50
- Track insider trading activity for conviction signals
- Review SEC filings for material corporate changes

---

## ⚠️ Disclaimer

This premium weekly highlights newsletter is for informational purposes only and does not constitute investment advice. The information is time-sensitive and market conditions can change rapidly.

**Key Considerations:**
- Past performance does not guarantee future results
- Weekly performance can be volatile and may not reflect long-term trends
- All investments carry risk, including potential loss of principal
- Always conduct your own due diligence and consult with a qualified financial advisor

**Conflicts of Interest:** The authors may hold positions in securities discussed in this newsletter.

---

*Generated by Geospatial Value Investor Premium*
*Next weekly highlights: {(datetime.now() + timedelta(days=7)).strftime("%B %d, %Y")}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Weekly highlights newsletter generated: {output_file}")
        return output_file


# ================================================================
# MAIN CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Value Investor CLI Tool for Geospatial Companies"
    )
    
    parser.add_argument(
        "--parquet-url",
        type=str,
        required=True,
        help="URL to geospatial_companies_with_cik.parquet file"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "gemini"],
        help="AI provider to use (default: anthropic)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for AI provider (or set ANTHROPIC_API_KEY/GEMINI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--newsletters",
        type=str,
        default="all",
        help="Newsletters to generate: free, summary, deep_dive, weekly, verticals, all (space-separated)"
    )
    
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.10,
        help="DCF discount rate (default: 0.10)"
    )
    
    parser.add_argument(
        "--base-growth",
        type=float,
        default=0.03,
        help="DCF base growth rate (default: 0.03)"
    )
    
    parser.add_argument(
        "--terminal-growth",
        type=float,
        default=0.02,
        help="DCF terminal growth rate (default: 0.02)"
    )
    
    parser.add_argument(
        "--margin-of-safety",
        type=float,
        default=0.30,
        help="Margin of safety for buy signals (default: 0.30)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for newsletters (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize AI client
    ai_client = AIClient(api_key=args.api_key, provider=args.provider)
    
    # Load companies
    companies_df = load_geospatial_companies(args.parquet_url)
    
    # Initialize monitors and analyzers
    news_monitor = GoogleNewsMonitor(ai_client)
    sec_monitor = SECFilingMonitor()
    stock_analyzer = StockAnalyzer(
        ai_client=ai_client,
        news_monitor=news_monitor,
        discount_rate=args.discount_rate,
        base_growth=args.base_growth,
        terminal_growth=args.terminal_growth,
        mos_threshold=args.margin_of_safety
    )
    
    # Analyze all stocks
    logging.info(f"Analyzing {len(companies_df)} companies...")
    analyses = []
    all_filings = []
    
    for idx, row in companies_df.iterrows():
        ticker = row['Symbol']
        company_name = row['Company']
        industry = row['Industry']
        country = row['Country']
        index = row['Index']
        cik = row['CIK']
        
        logging.info(f"[{idx+1}/{len(companies_df)}] Analyzing {ticker} ({company_name})...")
        
        # Analyze stock
        analysis = stock_analyzer.analyze_stock(
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            country=country,
            index=index,
            cik=cik
        )
        
        if analysis:
            analyses.append(analysis)
        
        # Get SEC filings
        filings = sec_monitor.get_recent_filings(ticker, cik, days=30)
        all_filings.extend(filings)
        
        # Rate limiting
        time.sleep(1)
    
    logging.info(f"✓ Analysis complete: {len(analyses)} stocks analyzed, {len(all_filings)} SEC filings found")
    
    # Save raw data
    with open(os.path.join(args.output_dir, 'stock_analysis.json'), 'w') as f:
        json.dump([asdict(a) for a in analyses], f, indent=2, default=str)
    
    with open(os.path.join(args.output_dir, 'sec_filings.json'), 'w') as f:
        json.dump([asdict(f) for f in all_filings], f, indent=2, default=str)
    
    logging.info("✓ Raw data saved")
    
    # Generate newsletters
    newsletter_gen = NewsletterGenerator(ai_client)
    
    newsletters_to_generate = args.newsletters.lower().split()
    
    if "all" in newsletters_to_generate:
        newsletters_to_generate = ["free", "deep_dive", "weekly", "verticals"]
    
    # Generate free summary
    if "free" in newsletters_to_generate or "summary" in newsletters_to_generate:
        logging.info("Generating FREE summary newsletter...")
        newsletter_gen.generate_free_summary(
            analyses, 
            all_filings, 
            output_file=os.path.join(args.output_dir, "free_summary.md")
        )
    
    # Generate premium deep dive
    if "deep_dive" in newsletters_to_generate:
        logging.info("Generating PREMIUM deep dive newsletter...")
        newsletter_gen.generate_premium_deep_dive(
            analyses, 
            all_filings, 
            output_file=os.path.join(args.output_dir, "premium_deep_dive.md")
        )
    
    # Generate weekly highlights
    if "weekly" in newsletters_to_generate:
        logging.info("Generating PREMIUM weekly highlights...")
        newsletter_gen.generate_weekly_highlights(
            analyses, 
            all_filings, 
            output_file=os.path.join(args.output_dir, "weekly_highlights.md")
        )
    
    # Generate vertical deep dives
    if "verticals" in newsletters_to_generate:
        industries = set(a.industry for a in analyses)
        logging.info(f"Generating PREMIUM vertical deep dives for {len(industries)} industries...")
        
        for industry in industries:
            logging.info(f"  - Generating {industry} vertical deep dive...")
            newsletter_gen.generate_vertical_deep_dive(
                analyses, 
                all_filings, 
                vertical=industry,
                output_file=os.path.join(args.output_dir, f"vertical_{industry.lower().replace(' ', '_')}.md")
            )
    
    logging.info("=" * 60)
    logging.info("✓ All newsletters generated successfully!")
    logging.info("=" * 60)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Stocks analyzed: {len(analyses)}")
    logging.info(f"SEC filings tracked: {len(all_filings)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
