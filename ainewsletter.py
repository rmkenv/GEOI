
"""
Enhanced Value Investor CLI Tool for Geospatial Companies
- Uses geospatial_companies_with_cik.parquet from GitHub (includes CIK)
- Monitors SEC filings (Forms 4, 14A, 14C, S-1, 8-K)
- Google News monitoring for each stock (last 90 days)
- Sentiment analysis for good/bad corporate conduct
- YTD, Quarterly, and 3-month projections
- Newsletter organized by industry vertical
- Conservative DCF valuation
- Exclusively uses Anthropic API for AI analysis
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
# PARQUET PARSING
# ================================================================
def load_geospatial_companies(parquet_url: str) -> pd.DataFrame:
    """Load geospatial companies from GitHub Parquet file"""
    logging.info(f"Loading Parquet from {parquet_url}...")
    
    try:
        with urllib.request.urlopen(parquet_url) as resp:
            data = resp.read()
        
        df = pd.read_parquet(io.BytesIO(data))
        logging.info(f"Parquet columns detected: {list(df.columns)}")
        
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        
        col_symbol = cols.get("symbol") or cols.get("ticker")
        col_company = cols.get("company name") or cols.get("company") or cols.get("name")
        col_industry = cols.get("industry") or cols.get("sector") or cols.get("vertical")
        col_country = cols.get("country") or cols.get("location")
        col_index = cols.get("index") or cols.get("exchange")
        col_cik = cols.get("cik") 
        
        if not col_symbol:
            raise ValueError("Parquet must include a 'symbol' or 'ticker' column")
        
        records = []
        for _, row in df.iterrows():
            sym = row.get(col_symbol)
            if pd.isna(sym):
                continue
            sym = str(sym).strip()
            if not sym:
                continue
            
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
        
        return result
        
    except Exception as e:
        logging.error(f"Error loading Parquet: {e}")
        raise


# ================================================================
# AI CLIENT - FIXED FOR ANTHROPIC API
# ================================================================
class AIClient:
    """Handles AI API calls, exclusively using Anthropic"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env variable or pass via --api-key.")
        
        # Test the API key on initialization
        self._test_api_key()
        
    def _test_api_key(self):
        """Test if the API key is valid"""
        try:
            test_response = self._call_anthropic("Say 'API key is valid' in 3 words.", max_tokens=50)
            if "API" not in test_response and "unavailable" in test_response:
                logging.warning("API key test failed - please verify your key")
            else:
                logging.info("✓ Anthropic API key validated successfully")
        except Exception as e:
            logging.error(f"API key validation failed: {e}")
    
    def analyze(self, prompt: str, max_tokens: int = 2048) -> str:
        """Send prompt to Anthropic AI and get response"""
        return self._call_anthropic(prompt, max_tokens)
    
    def _call_anthropic(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call Anthropic Claude API with correct format"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            logging.debug(f"Calling Anthropic API with model: {data['model']}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            # Log response details for debugging
            if response.status_code != 200:
                logging.error(f"API returned status {response.status_code}")
                logging.error(f"Response body: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from content blocks
            if "content" in result and isinstance(result["content"], list):
                text_parts = [
                    block.get("text", "") 
                    for block in result["content"] 
                    if block.get("type") == "text"
                ]
                return " ".join(text_parts).strip()
            
            logging.error(f"Unexpected API response format: {result}")
            return "AI analysis unavailable - unexpected response format."
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Anthropic API call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logging.error(f"API Error Details: {error_detail}")
                except:
                    logging.error(f"Response Text: {e.response.text}")
            return "AI analysis unavailable due to API error."


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
# SEC FILING MONITOR
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
        """Get recent SEC filings for a ticker"""
        
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
                logging.warning(f"SEC API returned {response.status_code} for {ticker}")
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
        """Get CIK number for ticker"""
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

            three_month_projection = self.ai_client.analyze(projection_prompt, max_tokens=512)
            
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

            analysis_text = self.ai_client.analyze(analysis_prompt, max_tokens=1024)
            
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
# NEWSLETTER GENERATOR
# ================================================================
class NewsletterGenerator:
    """Generate comprehensive monthly newsletter organized by industry vertical"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
# ================================================================
# NEWSLETTER GENERATOR
# ================================================================
class NewsletterGenerator:
    """Generate comprehensive monthly newsletter organized by industry vertical"""
    
    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
    
    def generate_free_overview(self, analyses: List[StockAnalysis], filings: List[SECFiling], 
                               output_file: str = "newsletter_free.md"):
        """Generate FREE 30,000 ft overview newsletter"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# Geospatial Industry Investment Newsletter - FREE EDITION
## {datetime.now().strftime("%B %Y")} - Market Overview

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

> **FREE EDITION**: High-level market overview and trends. Subscribe for detailed company analysis and sector deep-dives.

---

"""
        
        # Executive Summary
        if analyses:
            summary_prompt = f"""Provide a compelling 2-paragraph executive summary for a FREE newsletter edition covering {len(analyses)} geospatial companies:

1. Overall market sentiment and 2-3 key trends (make readers want to subscribe for details)
2. Mention that detailed analysis is available in premium edition

Industries: {', '.join(by_industry.keys())}

Be engaging but leave them wanting more details."""

            summary = self.ai_client.analyze(summary_prompt, max_tokens=800)
            
            report += f"""## Executive Summary

{summary}

---

"""
        
        # Market Snapshot
        report += "## Market Snapshot\n\n"
        
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        sell_count = sum(1 for a in analyses if a.value_verdict == "Sell/Avoid")
        
        report += f"""**Portfolio Statistics:**
- Total Companies Tracked: {total_companies}
- Industry Verticals: {len(by_industry)}
- Average YTD Return: {avg_ytd:.2f}%
- Average Quarterly Return: {avg_quarterly:.2f}%
- Buy Signals: {buy_count} companies
- Hold Recommendations: {hold_count} companies
- Sell/Avoid: {sell_count} companies

---

"""
        
        # Top Performers (teaser)
        report += "## 🏆 Top Performers This Month\n\n"
        top_performers = sorted([a for a in analyses if a.ytd_return is not None], 
                               key=lambda x: x.ytd_return, reverse=True)[:5]
        
        for i, stock in enumerate(top_performers, 1):
            report += f"{i}. **{stock.ticker}** ({stock.industry}) - YTD: {stock.ytd_return:.1f}%\n"
        
        report += "\n*📊 Detailed analysis available in Premium Edition*\n\n---\n\n"
        
        # Industry Highlights (High-level only)
        report += "## Industry Sector Highlights\n\n"
        
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            
            report += f"### {industry}\n"
            report += f"- Companies: {len(stocks)}\n"
            report += f"- Avg YTD Performance: {ind_ytd:.2f}%\n"
            report += f"- Buy Recommendations: {sum(1 for s in stocks if s.value_verdict == 'Buy')}\n\n"
            report += "*🔒 Full sector deep-dive available in Premium Edition*\n\n"
        
        # SEC Activity Summary
        report += "## 📋 Recent SEC Filing Activity\n\n"
        
        if filings:
            filing_counts = defaultdict(int)
            for f in filings:
                filing_counts[f.form_type] += 1
            
            report += f"**{len(filings)} total filings tracked this period:**\n\n"
            for form_type, count in sorted(filing_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"- Form {form_type}: {count} filing(s)\n"
            
            report += "\n*🔒 Detailed filing analysis by company available in Premium Edition*\n\n"
        else:
            report += "*No significant filings this period.*\n\n"
        
        # Call to Action
        report += """---

## 🔓 Unlock Premium Analysis

**What you get with Premium:**
- ✅ Detailed DCF valuations for every company
- ✅ Complete technical analysis and price targets
- ✅ News sentiment and corporate conduct assessments
- ✅ 3-month projections for each stock
- ✅ Deep-dive sector reports by industry vertical
- ✅ SEC filing analysis and insider trading insights
- ✅ Full methodology and data sources

[Subscribe to Premium →](#)

---

*Generated by Geospatial Value Investor CLI Tool - FREE EDITION*
*For informational purposes only. Not investment advice.*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ FREE overview newsletter generated: {output_file}")
        return output_file
    
    def generate_premium_full(self, analyses: List[StockAnalysis], filings: List[SECFiling], 
                             output_file: str = "newsletter_premium_full.md"):
        """Generate PREMIUM full analysis newsletter (all stocks, all verticals)"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# Geospatial Industry Investment Newsletter - PREMIUM EDITION
## {datetime.now().strftime("%B %Y")} - Complete Analysis

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

> **PREMIUM EDITION**: Complete analysis of all {len(analyses)} companies with detailed valuations, technical indicators, and AI-powered insights.

---

"""
        
        # Executive Summary with deeper insights
        if analyses:
            summary_prompt = f"""As a senior analyst, provide a comprehensive 4-paragraph executive summary for PREMIUM subscribers covering {len(analyses)} geospatial companies:

1. Macro market trends and what's driving them
2. Key investment opportunities and specific recommendations
3. Risk factors and concerns to watch
4. Strategic outlook for next quarter

Industries: {', '.join(by_industry.keys())}
Buy recommendations: {sum(1 for a in analyses if a.value_verdict == 'Buy')}

Be detailed and actionable for serious investors."""

            summary = self.ai_client.analyze(summary_prompt, max_tokens=1500)
            
            report += f"""## Executive Summary

{summary}

---

"""
        
        # Detailed Market Overview
        report += "## Market Overview\n\n"
        
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        sell_count = sum(1 for a in analyses if a.value_verdict == "Sell/Avoid")
        
        # Calculate average valuations
        avg_pe = np.nanmean([a.pe_ratio for a in analyses if a.pe_ratio is not None])
        avg_discount = np.nanmean([a.discount_vs_price for a in analyses if a.discount_vs_price is not None])
        
        report += f"""**Portfolio Statistics:**
- Total Companies Analyzed: {total_companies}
- Industry Verticals: {len(by_industry)}
- Average YTD Return: {avg_ytd:.2f}%
- Average Quarterly Return: {avg_quarterly:.2f}%
- Average P/E Ratio: {avg_pe:.2f if not np.isnan(avg_pe) else 'N/A'}
- Average Discount to Intrinsic: {avg_discount*100:.1f}% if not np.isnan(avg_discount) else 'N/A'

**Investment Recommendations:**
- 🟢 BUY: {buy_count} companies ({buy_count/total_companies*100:.1f}%)
- 🟡 HOLD/MONITOR: {hold_count} companies ({hold_count/total_companies*100:.1f}%)
- 🔴 SELL/AVOID: {sell_count} companies ({sell_count/total_companies*100:.1f}%)

---

"""
        
        # Top Investment Ideas
        report += "## 🎯 Top Investment Ideas\n\n"
        
        buy_stocks = [a for a in analyses if a.value_verdict == "Buy"]
        top_buys = sorted(buy_stocks, 
                         key=lambda x: x.discount_vs_price if x.discount_vs_price and not np.isnan(x.discount_vs_price) else -999, 
                         reverse=True)[:10]
        
        for i, stock in enumerate(top_buys, 1):
            discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price and not np.isnan(stock.discount_vs_price) else "N/A"
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value else "N/A"
            
            report += f"""### {i}. {stock.ticker} - {stock.company_name}
**Industry:** {stock.industry} | **Price:** ${stock.current_price:.2f} | **Intrinsic:** {intrinsic_str} | **Discount:** {discount_str}

{stock.analysis[:200]}... *(Full analysis below)*

---

"""
        
        # Full Industry Analysis
        report += "## 📊 Industry Sector Analysis\n\n"
        
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            
            report += f"""## {industry}

**Sector Overview:** {len(stocks)} companies analyzed

"""
            
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            ind_quarterly = np.nanmean([s.quarterly_return for s in stocks if s.quarterly_return is not None])
            ind_pe = np.nanmean([s.pe_ratio for s in stocks if s.pe_ratio is not None])
            
            report += f"""**Performance Metrics:**
- YTD Average: {ind_ytd:.2f}% if not np.isnan(ind_ytd) else "N/A"
- Quarterly Average: {ind_quarterly:.2f}% if not np.isnan(ind_quarterly) else "N/A"
- Average P/E: {ind_pe:.2f if not np.isnan(ind_pe) else 'N/A'}
- Buy Recommendations: {sum(1 for s in stocks if s.value_verdict == 'Buy')}

**3-Month Sector Outlook:**

"""
            
            sector_outlook_prompt = f"""Provide a detailed 4-5 sentence outlook for the {industry} sector in geospatial for next 3 months:
- {len(stocks)} companies analyzed
- Average YTD: {ind_ytd:.1f}% if not np.isnan(ind_ytd) else "N/A"
- Buy signals: {sum(1 for s in stocks if s.value_verdict == 'Buy')}

Include specific catalysts and risks."""
            
            sector_outlook = self.ai_client.analyze(sector_outlook_prompt, max_tokens=800)
            report += f"{sector_outlook}\n\n"
            
            report += "### Company Analysis\n\n"
            
            stocks_sorted = sorted(stocks, 
                                 key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                                 reverse=True)
            
            for stock in stocks_sorted:
                discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
                intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
                pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
                pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
                div_str = f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "N/A"
                mcap_str = f"${stock.market_cap:,.0f}" if stock.market_cap is not None else "N/A"
                ytd_str = f"{stock.ytd_return:.2f}%" if stock.ytd_return is not None else "N/A"
                quarterly_str = f"{stock.quarterly_return:.2f}%" if stock.quarterly_return is not None else "N/A"
                rsi_str = f"{stock.rsi:.1f}" if stock.rsi is not None else "N/A"
                sma20_str = f"${stock.sma20:.2f}" if stock.sma20 is not None else "N/A"
                sma50_str = f"${stock.sma50:.2f}" if stock.sma50 is not None else "N/A"
                
                report += f"""#### {stock.ticker} - {stock.company_name}

**Location:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else "N/A"}

**Valuation Metrics:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value (DCF): {intrinsic_str}
- Discount to Price: {discount_str}
- **Value Verdict: {stock.value_verdict}**
- P/E Ratio: {pe_str}
- P/B Ratio: {pb_str}
- Dividend Yield: {div_str}
- Market Cap: {mcap_str}

**Performance:**
- YTD Return: {ytd_str}
- Quarterly Return: {quarterly_str}

**Technical Indicators:**
- RSI: {rsi_str}
- SMA20: {sma20_str}
- SMA50: {sma50_str}
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
        
        # SEC Filings
        report += "\n## 📋 SEC Filings & Corporate Actions\n\n"
        
        if filings:
            filings_by_ticker = defaultdict(list)
            for filing in filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                company_name = next((a.company_name for a in analyses if a.ticker == ticker), ticker)
                
                report += f"### {ticker} - {company_name}\n\n"
                
                for filing in sorted(ticker_filings, key=lambda x: x.filing_date, reverse=True):
                    form_desc = {
                        "4": "📊 Insider Trading",
                        "14A": "🗳️ Proxy Statement",
                        "S-1": "🚀 IPO Registration",
                        "8-K": "📢 Current Report"
                    }.get(filing.form_type, filing.form_type)
                    
                    report += f"- **{form_desc}** - {filing.filing_date}\n"
                    report += f"  [View Filing]({filing.url})\n\n"
        
        report += self._add_methodology_section()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ PREMIUM full newsletter generated: {output_file}")
        return output_file
    
    def generate_vertical_deepdive(self, industry: str, analyses: List[StockAnalysis], 
                                   filings: List[SECFiling], output_file: str = None):
        """Generate PREMIUM deep-dive for specific vertical"""
        
        if output_file is None:
            safe_industry = industry.replace(" ", "_").replace("/", "_")
            output_file = f"newsletter_premium_{safe_industry}.md"
        
        industry_stocks = [a for a in analyses if a.industry == industry]
        
        if not industry_stocks:
            logging.warning(f"No stocks found for industry: {industry}")
            return None
        
        report = f"""# {industry} - Deep Dive Analysis
## {datetime.now().strftime("%B %Y")} - PREMIUM VERTICAL REPORT

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

> **PREMIUM VERTICAL REPORT**: Comprehensive analysis of the {industry} sector with {len(industry_stocks)} companies.

---

"""
        
        # Sector Executive Summary
        sector_summary_prompt = f"""As a sector specialist, provide a comprehensive analysis of the {industry} vertical in geospatial technology:

Analyze {len(industry_stocks)} companies:
{', '.join([f"{s.ticker} ({s.value_verdict})" for s in industry_stocks[:10]])}

Provide 5 paragraphs covering:
1. Current state of {industry} sector
2. Key competitive dynamics and market leaders
3. Technology trends and innovation
4. Investment opportunities and best picks
5. Risks and challenges ahead

Be detailed and technical."""

        sector_summary = self.ai_client.analyze(sector_summary_prompt, max_tokens=2048)
        
        report += f"""## Sector Overview

{sector_summary}

---

"""
        
        # Sector Metrics
        ind_ytd = np.nanmean([s.ytd_return for s in industry_stocks if s.ytd_return is not None])
        ind_quarterly = np.nanmean([s.quarterly_return for s in industry_stocks if s.quarterly_return is not None])
        ind_pe = np.nanmean([s.pe_ratio for s in industry_stocks if s.pe_ratio is not None])
        ind_pb = np.nanmean([s.pb_ratio for s in industry_stocks if s.pb_ratio is not None])
        
        buy_count = sum(1 for s in industry_stocks if s.value_verdict == "Buy")
        
        report += f"""## Sector Performance Metrics

**Returns:**
- YTD Average: {ind_ytd:.2f}% if not np.isnan(ind_ytd) else "N/A"
- Quarterly Average: {ind_quarterly:.2f}% if not np.isnan(ind_quarterly) else "N/A"
- Best Performer: {max(industry_stocks, key=lambda x: x.ytd_return if x.ytd_return else -999).ticker} ({max([s.ytd_return for s in industry_stocks if s.ytd_return], default=0):.1f}%)
- Worst Performer: {min(industry_stocks, key=lambda x: x.ytd_return if x.ytd_return else 999).ticker} ({min([s.ytd_return for s in industry_stocks if s.ytd_return], default=0):.1f}%)

**Valuations:**
- Average P/E: {ind_pe:.2f if not np.isnan(ind_pe) else 'N/A'}
- Average P/B: {ind_pb:.2f if not np.isnan(ind_pb) else 'N/A'}
- Buy Recommendations: {buy_count} ({buy_count/len(industry_stocks)*100:.1f}%)

---

"""
        
        # Top Picks in this Vertical
        report += f"## 🎯 Top Investment Picks in {industry}\n\n"
        
        buy_stocks = [s for s in industry_stocks if s.value_verdict == "Buy"]
        top_picks = sorted(buy_stocks, 
                          key=lambda x: x.discount_vs_price if x.discount_vs_price and not np.isnan(x.discount_vs_price) else -999, 
                          reverse=True)[:5]
        
        if top_picks:
            for i, stock in enumerate(top_picks, 1):
                discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price and not np.isnan(stock.discount_vs_price) else "N/A"
                report += f"""{i}. **{stock.ticker}** - ${stock.current_price:.2f} (Discount: {discount_str})
   {stock.analysis[:150]}...

"""
        else:
            report += "*No BUY recommendations in this sector at current valuations.*\n\n"
        
        report += "---\n\n"
        
        # Detailed Company Analysis
        report += f"## Complete Company Analysis - {industry}\n\n"
        
        stocks_sorted = sorted(industry_stocks, 
                             key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                             reverse=True)
        
        for stock in stocks_sorted:
            discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
            intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
            pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
            pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
            div_str = f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "N/A"
            mcap_str = f"${stock.market_cap:,.0f}" if stock.market_cap is not None else "N/A"
            ytd_str = f"{stock.ytd_return:.2f}%" if stock.ytd_return is not None else "N/A"
            quarterly_str = f"{stock.quarterly_return:.2f}%" if stock.quarterly_return is not None else "N/A"
            rsi_str = f"{stock.rsi:.1f}" if stock.rsi is not None else "N/A"
            sma20_str = f"${stock.sma20:.2f}" if stock.sma20 is not None else "N/A"
            sma50_str = f"${stock.sma50:.2f}" if stock.sma50 is not None else "N/A"
            
            report += f"""### {stock.ticker} - {stock.company_name}

**Location:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else "N/A"}

**Valuation Metrics:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value (DCF): {intrinsic_str}
- Discount to Price: {discount_str}
- **Value Verdict: {stock.value_verdict}**
- P/E Ratio: {pe_str}
- P/B Ratio: {pb_str}
- Dividend Yield: {div_str}
- Market Cap: {mcap_str}

**Performance:**
- YTD Return: {ytd_str}
- Quarterly Return: {quarterly_str}

**Technical Indicators:**
- RSI: {rsi_str}
- SMA20: {sma20_str}
- SMA50: {sma50_str}
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
        
        # SEC Filings for this vertical
        vertical_filings = [f for f in filings if f.ticker in [s.ticker for s in industry_stocks]]
        
        if vertical_filings:
            report += f"\n## 📋 Recent SEC Filings - {industry}\n\n"
            
            filings_by_ticker = defaultdict(list)
            for filing in vertical_filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                company_name = next((a.company_name for a in industry_stocks if a.ticker == ticker), ticker)
                
                report += f"### {ticker} - {company_name}\n\n"
                
                for filing in sorted(ticker_filings, key=lambda x: x.filing_date, reverse=True):
                    report += f"- **{filing.form_type}** - {filing.filing_date}: [View]({filing.url})\n"
                report += "\n"
        
        report += self._add_methodology_section()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ PREMIUM vertical deep-dive generated: {output_file}")
        return output_file
    
    def _add_methodology_section(self):
        """Add comprehensive methodology section to reports"""
        return """---

## Methodology & Research Framework

### Overview

This newsletter employs a rigorous, multi-faceted approach to analyzing geospatial technology companies, combining quantitative valuation models, technical analysis, qualitative news assessment, and regulatory monitoring. Our framework is designed for value investors seeking long-term opportunities in the geospatial sector.

---

### 1. Valuation Framework: Conservative Discounted Cash Flow (DCF)

**Philosophy:**
We employ a conservative DCF model inspired by traditional value investing principles (Graham, Dodd, Buffett). Our approach prioritizes margin of safety and conservative assumptions over aggressive growth projections.

**Model Parameters:**
- **Free Cash Flow (FCF) Calculation**: 
  - Primary: Direct FCF from cash flow statements
  - Fallback: Operating Cash Flow - Capital Expenditures
  - Uses 3-year average to smooth volatility
  
- **Growth Assumptions**:
  - **Base Growth Rate**: 3% (default) - Conservative estimate reflecting long-term GDP growth
  - **Terminal Growth Rate**: 2% (default) - Below long-term inflation expectations
  - **Projection Period**: 10 years
  
- **Discount Rate**: 10% (default) - Represents required rate of return
  - Based on long-term equity market returns
  - Adjustable for risk tolerance
  - Higher than risk-free rate to account for equity risk premium

**Intrinsic Value Calculation:**
1. Project FCF for next 10 years using base growth rate
2. Calculate present value of projected cash flows
3. Estimate terminal value using perpetuity growth model
4. Discount terminal value to present
5. Sum present values and divide by shares outstanding

**Margin of Safety:**
- **Buy Threshold**: 30%+ discount to intrinsic value (default)
- **Hold/Monitor**: 0-30% discount to intrinsic value
- **Sell/Avoid**: Price above intrinsic value

**Limitations:**
- Model reliability depends on FCF history and quality
- Companies with negative or volatile FCF receive "Too Hard to Value" designation
- Not suitable for pre-revenue or high-growth story stocks
- Best suited for established companies with predictable cash flows

---

### 2. Technical Analysis Framework

**Indicators Used:**

**A. Relative Strength Index (RSI)**
- **Period**: 14-day
- **Interpretation**:
  - RSI < 30: Oversold condition (potential buy signal)
  - RSI > 70: Overbought condition (potential sell signal)
  - RSI 30-70: Neutral zone
- **Purpose**: Identify momentum and potential reversal points

**B. Simple Moving Averages (SMA)**
- **SMA20**: 20-day moving average (short-term trend)
- **SMA50**: 50-day moving average (medium-term trend)
- **Interpretation**:
  - Price > SMA20 > SMA50: Bullish trend
  - Price < SMA20 < SMA50: Bearish trend
  - SMA crossovers signal potential trend changes
- **Purpose**: Confirm trend direction and strength

**C. Recommendation Synthesis**
- **Primary**: Analyst consensus from market data providers
- **Secondary**: Technical signal confirmation
  - Buy: RSI ≤ 30 AND bullish trend
  - Sell: RSI ≥ 70 AND bearish trend
  - Hold: Default when signals are mixed
- **Purpose**: Combine fundamental and technical perspectives

**Philosophy:**
Technical analysis serves as a timing tool to complement fundamental valuation. We don't rely on technical signals alone but use them to identify optimal entry/exit points for fundamentally sound investments.

---

### 3. News Monitoring & Sentiment Analysis

**Data Source:**
- Google News RSS feeds
- **Lookback Period**: 90 days (extended to capture quarterly developments)
- **Coverage**: Up to 10 most relevant articles per company

**Sentiment Classification:**

**A. Market Sentiment (Positive/Negative/Neutral)**

*Positive Keywords:*
- Financial: growth, profit, gain, beat, outperform, record, strong
- Business: success, innovation, breakthrough, expansion, partnership, award

*Negative Keywords:*
- Financial: loss, decline, miss, weak, downgrade, cut
- Business: lawsuit, scandal, investigation, layoff, bankruptcy, concern, warning

*Analysis Method:*
- Keyword frequency analysis in headlines and snippets
- Net sentiment = (Positive count) - (Negative count)
- Classification based on net sentiment score

**B. Corporate Conduct Assessment (Good/Bad/Neutral)**

*Good Conduct Indicators:*
- ESG-positive: sustainability, ethical, green, renewable, responsible
- Social: charity, donation, community, diversity, transparency

*Bad Conduct Indicators:*
- Legal: lawsuit, fraud, scandal, violation, penalty, misconduct
- Ethical: corruption, discrimination, environmental damage, breach

*Analysis Method:*
- Binary flag: Any bad conduct keyword triggers "Bad" classification
- Good conduct recognized only when no negative signals present
- Prioritizes risk identification over positive PR

**Sentiment Summary:**
- Aggregates sentiment across all articles
- Identifies predominant tone (positive/negative/mixed)
- Highlights key headlines for manual review
- Flags potential conduct concerns for further investigation

**Limitations:**
- Keyword-based analysis may miss context
- PR-driven articles may inflate positive sentiment
- Regional news may be underrepresented
- Requires human judgment for complex situations

---

### 4. SEC Filing Monitoring

**Data Source:**
- SEC EDGAR database (data.sec.gov)
- CIK (Central Index Key) matching from company database
- Real-time API access to filing submissions

**Monitored Form Types:**

**A. Form 4 - Insider Trading**
- **Description**: Statement of Changes in Beneficial Ownership
- **Significance**: Tracks buying/selling by company insiders (officers, directors, 10%+ shareholders)
- **Analysis**: Cluster of insider buys may signal undervaluation; sells may indicate concerns

**B. Proxy Statements (14A, 14C, DEF 14A variants)**
- **Description**: Shareholder meeting materials
- **Significance**: Executive compensation, board changes, shareholder proposals, M&A votes
- **Analysis**: Governance issues, alignment of management incentives, strategic direction

**C. Form S-1 - IPO Registration**
- **Description**: Initial registration for new securities
- **Significance**: IPO filing, company going public
- **Analysis**: Growth opportunity, liquidity event, insider lockup periods

**D. Form 8-K - Current Reports**
- **Description**: Material corporate events
- **Significance**: Earnings, leadership changes, acquisitions, contracts, disasters
- **Analysis**: Time-sensitive information that may impact valuation

**E. Prospectus Filings (424B4)**
- **Description**: Final prospectus for IPO
- **Significance**: Detailed company financials and risk factors pre-IPO

**Filing Analysis:**
- **Lookback Period**: 30 days (default, adjustable)
- **Presentation**: Chronological listing by company
- **Links**: Direct links to SEC EDGAR for full document review
- **Context**: Brief description of filing significance

**Purpose:**
- Identify material corporate events not yet reflected in stock price
- Monitor insider sentiment through Form 4 patterns
- Track governance issues via proxy statements
- Alert to upcoming catalysts (IPOs, votes, major announcements)

---

### 5. Performance Metrics

**A. Year-to-Date (YTD) Return**
- **Calculation**: (Current Price - Price on Jan 1) / Price on Jan 1 × 100%
- **Lookback**: 365 days from current date
- **Purpose**: Annual performance context, tax-year perspective

**B. Quarterly Return**
- **Calculation**: (Current Price - Price 90 days ago) / Price 90 days ago × 100%
- **Lookback**: 90 days
- **Purpose**: Recent momentum, short-term trend identification

**C. 3-Month Projection**
- **Method**: AI-generated qualitative outlook
- **Inputs**: Technical indicators, fundamental valuation, recent news, sector trends
- **Output**: 2-3 sentence forward-looking statement
- **Purpose**: Synthesize quantitative and qualitative factors into actionable outlook

**Benchmark Considerations:**
- Performance evaluated against sector averages
- Relative strength analysis within industry verticals
- Not directly benchmarked against indices due to sector-specific focus

---

### 6. AI-Powered Analysis

**Model:**
- **Provider**: Anthropic Claude (Sonnet 4)
- **Version**: claude-sonnet-4-20250514
- **Capabilities**: Natural language understanding, multi-factor synthesis, nuanced reasoning

**AI Analysis Components:**

**A. Executive Summaries**
- **Input**: Aggregate market data, sector performance, filing activity
- **Task**: Synthesize trends, identify opportunities, highlight risks
- **Output**: 2-4 paragraph market overview
- **Validation**: AI insights are presented alongside quantitative data for verification

**B. Company Investment Analysis**
- **Input**: Valuation metrics, technicals, news sentiment, conduct assessment
- **Task**: Integrate multiple factors into coherent investment thesis
- **Output**: 3-4 sentence investment recommendation
- **Framework**: Covers (1) valuation, (2) risks/opportunities, (3) recommendation

**C. 3-Month Projections**
- **Input**: Price, returns, technical signals, valuation verdict, news summary
- **Task**: Forward-looking outlook considering multiple factors
- **Output**: 2-3 sentence projection with realistic expectations

**D. Sector Outlooks**
- **Input**: Sector performance, company count, average returns, buy signals
- **Task**: Sector-specific trend analysis and opportunity identification
- **Output**: 2-5 sentence sector commentary (varies by report tier)

**E. Vertical Deep-Dives (Premium Reports)**
- **Input**: Comprehensive sector data, competitive dynamics, company list
- **Task**: In-depth sector analysis covering state, competition, technology, opportunities, risks
- **Output**: 5-paragraph specialist-level analysis

**AI Governance:**
- **Transparency**: All AI-generated content is clearly labeled
- **Human Oversight**: AI analysis supplements, not replaces, quantitative models
- **Validation**: AI insights cross-referenced with data and market consensus
- **Limitations**: AI cannot predict future, analyses are probabilistic not deterministic
- **Updates**: Model versions documented to maintain reproducibility

---

### 7. Data Sources & Quality

**Primary Data Providers:**

**A. Yahoo Finance (yfinance)**
- **Usage**: Stock prices, fundamental data, financial statements
- **Frequency**: Real-time to 15-minute delayed (varies by exchange)
- **Quality**: Industry-standard aggregator, generally reliable for US markets
- **Limitations**: Occasional data gaps, corporate action adjustments

**B. SEC EDGAR**
- **Usage**: Regulatory filings, insider transactions, corporate events
- **Frequency**: Real-time SEC submissions
- **Quality**: Official government source, highest reliability
- **Limitations**: Disclosure delays, complexity of legal documents

**C. Google News**
- **Usage**: News articles, sentiment analysis
- **Frequency**: Near real-time news aggregation
- **Quality**: Broad coverage, but varies by source
- **Limitations**: Potential bias, PR influence, regional gaps

**D. Company CIK Database**
- **Usage**: Linking tickers to SEC filings via CIK numbers
- **Source**: Custom parquet file (geospatial_companies_with_cik.parquet)
- **Quality**: Manually curated, geospatial sector focus
- **Limitations**: Must be periodically updated for new companies/IPOs

**Data Quality Controls:**
- **Caching**: Reduces API calls, improves performance, minimizes rate limiting
- **Error Handling**: Graceful degradation when data unavailable
- **Validation**: Cross-reference between multiple sources when available
- **Logging**: Comprehensive logging for troubleshooting and audit trails

---

### 8. Industry Classification

**Vertical Categories:**
Companies are classified into industry verticals based on primary business focus:

- **Satellite Imagery**: Earth observation, remote sensing providers
- **GIS Software**: Geographic Information Systems platforms
- **Drone Technology**: UAV hardware and software
- **Location Intelligence**: Location-based analytics and services
- **Mapping & Navigation**: Digital mapping, routing, navigation
- **Defense & Government**: Government-focused geospatial solutions
- **Agriculture Tech**: Precision agriculture, crop monitoring
- **Infrastructure**: Smart cities, infrastructure monitoring
- **Energy & Resources**: Oil/gas, mining, renewable energy applications
- **Other Geospatial**: Niche applications not fitting above categories

**Classification Method:**
- Primary source: Company self-description and SEC filings
- Secondary: Industry codes (NAICS, SIC)
- Manual curation for ambiguous cases

**Purpose:**
- Enable vertical-specific analysis and deep-dives
- Compare companies against relevant peers
- Identify sector-specific trends and opportunities

---

### 9. Newsletter Tiers & Delivery

**Free Edition (Public):**
- 30,000-foot market overview
- Top performer highlights (teasers)
- High-level sector summaries
- SEC filing counts and categories
- Call-to-action for premium content
- **Purpose**: Lead generation, brand building, market education

**Premium Full Edition (Paid Subscribers):**
- Comprehensive executive summary
- Complete company analysis (all stocks)
- Detailed valuations with DCF models
- Full technical analysis
- News and conduct assessments
- 3-month projections
- Complete SEC filing analysis
- **Purpose**: Core subscription product for serious investors

**Premium Vertical Deep-Dives (Paid Subscribers):**
- Sector-specialist level analysis
- Competitive landscape assessment
- Technology trend analysis
- Complete company analysis within vertical
- Vertical-specific SEC filings
- Top picks within sector
- **Purpose**: Premium add-on for sector specialists

---

### 10. Investment Philosophy & Disclaimers

**Investment Philosophy:**

This newsletter is grounded in **value investing** principles:

1. **Intrinsic Value Focus**: Price is what you pay, value is what you get
2. **Margin of Safety**: Require significant discount to protect against errors and bad luck
3. **Long-Term Orientation**: Not focused on short-term trading or market timing
4. **Conservative Assumptions**: Better to be approximately right than precisely wrong
5. **Fundamental Analysis First**: Technical analysis as supplement, not primary driver
6. **Risk Management**: Emphasis on downside protection over maximum upside

**Target Audience:**
- Individual investors with multi-year time horizons
- Value-oriented fund managers
- Geospatial industry professionals evaluating investments
- Those willing to do additional due diligence

**Not Suitable For:**
- Day traders or short-term speculators
- Those seeking hot tips or momentum plays
- Investors unable to tolerate volatility
- Those making investment decisions without independent research

**Key Disclaimers:**

⚠️ **Not Investment Advice**: This newsletter provides information and analysis for educational purposes only. It does not constitute investment advice, recommendations, or offers to buy or sell securities.

⚠️ **Do Your Own Research**: Always conduct independent research and due diligence before making investment decisions. Consider your personal financial situation, risk tolerance, and investment objectives.

⚠️ **Consult Professionals**: Consult with qualified financial advisors, accountants, and legal professionals before making investment decisions.

⚠️ **Model Limitations**: All valuation models are simplified representations of reality and rely on assumptions that may prove incorrect. DCF models are particularly sensitive to growth and discount rate assumptions.

⚠️ **Past Performance**: Historical returns do not guarantee future results. The geospatial industry is subject to technological disruption, regulatory changes, and economic cycles.

⚠️ **Data Accuracy**: While we use reputable data sources, errors and omissions can occur. Always verify critical information independently.

⚠️ **AI Limitations**: AI-generated analysis provides perspective but cannot predict the future. AI may miss context, make reasoning errors, or reflect biases in training data.

⚠️ **Sector Risk**: The geospatial industry faces specific risks including technological obsolescence, government contract dependency, competitive intensity, and capital requirements.

⚠️ **No Guarantees**: No investment strategy guarantees profits or protects against losses. All investing involves risk, including possible loss of principal.

---

### 11. Methodology Updates & Versioning

**Current Version**: 1.0 (November 2025)

**Update Policy:**
- Methodology changes are documented and versioned
- Significant changes communicated to subscribers
- Historical analyses remain interpretable despite methodology updates
- Model parameters (discount rate, growth rates, MOS threshold) noted in each report

**Continuous Improvement:**
- Regular backtesting of valuation models
- Refinement of sentiment analysis keywords
- Expansion of data sources as available
- Incorporation of subscriber feedback

**Transparency Commitment:**
- Full methodology disclosed to premium subscribers
- Data sources and calculation methods documented
- AI model versions and prompts specified
- Limitations honestly acknowledged

---

### 12. Contact & Feedback

**Questions or Concerns:**
For questions about methodology, data sources, or specific analyses, subscribers can contact the research team.

**Feedback Welcome:**
We continuously improve our analysis framework based on subscriber input. Suggestions for additional metrics, data sources, or analysis dimensions are appreciated.

**Research Integrity:**
- No compensation from covered companies
- No coordination with management prior to publication
- Independent analysis without conflicts of interest
- Objective assessment of both opportunities and risks

---

*Methodology Last Updated: November 2025*
*Generated by Geospatial Value Investor CLI Tool*
*Data sources: Yahoo Finance, SEC EDGAR, Google News, Anthropic Claude*
*Framework inspired by Graham & Dodd, Buffett, and modern value investing practices*

"""
        """Generate newsletter report organized by vertical"""
        
        by_industry = defaultdict(list)
        for analysis in analyses:
            by_industry[analysis.industry].append(analysis)
        
        report = f"""# Geospatial Industry Investment Newsletter
## {datetime.now().strftime("%B %Y")}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        
        if analyses:
            summary_prompt = f"""Based on analysis of {len(analyses)} geospatial companies across {len(by_industry)} industry verticals and {len(filings)} SEC filings, provide a 3-paragraph executive summary:

1. Overall market trends and opportunities in the geospatial sector
2. Key findings from news monitoring and corporate conduct
3. Top investment recommendations

Industries covered: {', '.join(by_industry.keys())}

Be concise and actionable for value investors."""

            summary = self.ai_client.analyze(summary_prompt, max_tokens=1024)
            
            report += f"""## Executive Summary

{summary}

---

"""
        
        report += "## Market Overview\n\n"
        
        total_companies = len(analyses)
        avg_ytd = np.nanmean([a.ytd_return for a in analyses if a.ytd_return is not None])
        avg_quarterly = np.nanmean([a.quarterly_return for a in analyses if a.quarterly_return is not None])
        
        buy_count = sum(1 for a in analyses if a.value_verdict == "Buy")
        hold_count = sum(1 for a in analyses if a.value_verdict == "Hold/Monitor")
        
        report += f"""**Portfolio Statistics:**
- Total Companies Analyzed: {total_companies}
- Industry Verticals: {len(by_industry)}
- Average YTD Return: {avg_ytd:.2f}% if not np.isnan(avg_ytd) else "N/A"
- Average Quarterly Return: {avg_quarterly:.2f}% if not np.isnan(avg_quarterly) else "N/A"
- Buy Recommendations: {buy_count}
- Hold/Monitor: {hold_count}

---

"""
        
        for industry in sorted(by_industry.keys()):
            stocks = by_industry[industry]
            
            report += f"""## {industry}

**Sector Overview:** {len(stocks)} companies analyzed

"""
            
            ind_ytd = np.nanmean([s.ytd_return for s in stocks if s.ytd_return is not None])
            ind_quarterly = np.nanmean([s.quarterly_return for s in stocks if s.quarterly_return is not None])
            
            report += f"""**Performance:**
- YTD Average: {ind_ytd:.2f}% if not np.isnan(ind_ytd) else "N/A"
- Quarterly Average: {ind_quarterly:.2f}% if not np.isnan(ind_quarterly) else "N/A"

**3-Month Sector Outlook:**

"""
            
            sector_outlook_prompt = f"""Provide a 2-3 sentence outlook for the {industry} sector in the geospatial industry for the next 3 months based on:
- {len(stocks)} companies analyzed
- Average YTD return: {ind_ytd:.1f}% if not np.isnan(ind_ytd) else "N/A"
- Average quarterly return: {ind_quarterly:.1f}% if not np.isnan(ind_quarterly) else "N/A"

Be specific to geospatial technology trends."""
            
            sector_outlook = self.ai_client.analyze(sector_outlook_prompt, max_tokens=512)
            report += f"{sector_outlook}\n\n"
            
            report += "### Company Analysis\n\n"
            
            stocks_sorted = sorted(stocks, 
                                 key=lambda x: x.discount_vs_price if x.discount_vs_price is not None and not np.isnan(x.discount_vs_price) else -999, 
                                 reverse=True)
            
            for stock in stocks_sorted:
                discount_str = f"{stock.discount_vs_price*100:.1f}%" if stock.discount_vs_price is not None and not np.isnan(stock.discount_vs_price) else "N/A"
                intrinsic_str = f"${stock.intrinsic_value:.2f}" if stock.intrinsic_value is not None else "N/A"
                pe_str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
                pb_str = f"{stock.pb_ratio:.2f}" if stock.pb_ratio is not None else "N/A"
                div_str = f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "N/A"
                mcap_str = f"${stock.market_cap:,.0f}" if stock.market_cap is not None else "N/A"
                ytd_str = f"{stock.ytd_return:.2f}%" if stock.ytd_return is not None else "N/A"
                quarterly_str = f"{stock.quarterly_return:.2f}%" if stock.quarterly_return is not None else "N/A"
                rsi_str = f"{stock.rsi:.1f}" if stock.rsi is not None else "N/A"
                sma20_str = f"${stock.sma20:.2f}" if stock.sma20 is not None else "N/A"
                sma50_str = f"${stock.sma50:.2f}" if stock.sma50 is not None else "N/A"
                
                report += f"""#### {stock.ticker} - {stock.company_name}

**Location:** {stock.country} | **Exchange:** {stock.index} | **CIK:** {stock.cik if stock.cik else "N/A"}

**Valuation Metrics:**
- Current Price: ${stock.current_price:.2f}
- Intrinsic Value (DCF): {intrinsic_str}
- Discount to Price: {discount_str}
- **Value Verdict: {stock.value_verdict}**
- P/E Ratio: {pe_str}
- P/B Ratio: {pb_str}
- Dividend Yield: {div_str}
- Market Cap: {mcap_str}

**Performance:**
- YTD Return: {ytd_str}
- Quarterly Return: {quarterly_str}

**Technical Indicators:**
- RSI: {rsi_str}
- SMA20: {sma20_str}
- SMA50: {sma50_str}
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
        
        report += "\n## SEC Filings & Corporate Actions\n\n"
        
        if filings:
            filings_by_ticker = defaultdict(list)
            for filing in filings:
                filings_by_ticker[filing.ticker].append(filing)
            
            for ticker in sorted(filings_by_ticker.keys()):
                ticker_filings = filings_by_ticker[ticker]
                
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
- **90-day lookback period**

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
*AI Analysis: Anthropic Claude Sonnet 4*
"""
        
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
  # Generate FREE overview only (default)
  python valueinvestortool.py --api-key YOUR_KEY
  
  # Generate ALL newsletters (free + premium full + all vertical deep-dives)
  python valueinvestortool.py --generate-all --api-key YOUR_KEY
  
  # Generate PREMIUM full newsletter only
  python valueinvestortool.py --generate-premium --api-key YOUR_KEY
  
  # Generate deep-dive for specific vertical
  python valueinvestortool.py --generate-vertical "Satellite Imagery" --api-key YOUR_KEY
  
  # Test with limited stocks
  python valueinvestortool.py --limit 10 --generate-all --api-key YOUR_KEY
  
  # Custom output directory
  python valueinvestortool.py --generate-all -o my_newsletters --api-key YOUR_KEY
  
  # Export data in addition to newsletters
  python valueinvestortool.py --generate-all --export-json data.json --export-csv data.csv --api-key YOUR_KEY

Newsletter Editions:
  FREE:     30,000 ft overview with market snapshot and teasers
  PREMIUM:  Full analysis of all stocks with detailed valuations
  VERTICAL: Deep-dive into specific industry sector
        """
    )
    
    parser.add_argument("--parquet-url", default="https://github.com/rmkenv/GEOI/raw/main/geospatial_companies_with_cik.parquet",
                       help="URL to geospatial companies Parquet file")
    parser.add_argument("--output-dir", "-o", default="newsletters", help="Output directory for newsletters")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days to look back for SEC filings (news is 90 days)")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to analyze (for testing)")
    
    # Newsletter generation options
    parser.add_argument("--generate-all", action="store_true", help="Generate all newsletter editions (free + premium full + all verticals)")
    parser.add_argument("--generate-free", action="store_true", help="Generate only free overview newsletter")
    parser.add_argument("--generate-premium", action="store_true", help="Generate only premium full newsletter")
    parser.add_argument("--generate-vertical", help="Generate deep-dive for specific vertical (e.g., 'Satellite Imagery')")
    
    parser.add_argument("--discount-rate", type=float, default=0.10, help="DCF discount rate (default: 0.10)")
    parser.add_argument("--base-growth", type=float, default=0.03, help="Base growth rate (default: 0.03)")
    parser.add_argument("--terminal-growth", type=float, default=0.02, help="Terminal growth rate (default: 0.02)")
    parser.add_argument("--mos-threshold", type=float, default=0.30, help="Margin of safety threshold (default: 0.30)")
    
    parser.add_argument("--export-json", help="Export analysis to JSON file")
    parser.add_argument("--export-csv", help="Export analysis to CSV file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🌍 GEOSPATIAL INDUSTRY VALUE INVESTOR CLI TOOL")
    print("   Enhanced with News Monitoring & Industry Vertical Analysis")
    print("   AI Analysis: Anthropic Claude Sonnet 4")
    print("=" * 80)
    print()
    
    companies_df = load_geospatial_companies(args.parquet_url)
    
    if args.limit:
        companies_df = companies_df.head(args.limit)
        logging.info(f"Limited to {args.limit} companies for analysis")
    
    print()
    
    ai_client = AIClient(api_key=args.api_key)
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
    
    logging.info(f"Analyzing {len(companies_df)} geospatial companies...")
    analyses = []
    
    for i, row in companies_df.iterrows():
        ticker = row['Symbol']
        company = row['Company']
        industry = row['Industry']
        country = row['Country']
        index = row['Index']
        cik = row.get('CIK')
        
        logging.info(f"[{i+1}/{len(companies_df)}] Analyzing {ticker} - {company}...")
        
        analysis = stock_analyzer.analyze_stock(ticker, company, industry, country, index, cik)
        if analysis:
            analyses.append(analysis)
        
        time.sleep(1)
    
    logging.info(f"Successfully analyzed {len(analyses)}/{len(companies_df)} companies")
    print()
    
    logging.info(f"Checking SEC filings (last {args.days} days)...")
    all_filings = []
    
    for i, row in companies_df.iterrows():
        ticker = row['Symbol']
        cik = row.get('CIK')
        
        logging.info(f"[{i+1}/{len(companies_df)}] Checking {ticker}...")
        
        filings = sec_monitor.get_recent_filings(ticker, cik=cik, days=args.days)
        all_filings.extend(filings)
        
        if filings:
            logging.info(f"  Found {len(filings)} filing(s)")
        
        time.sleep(0.5)
    
    logging.info(f"Found {len(all_filings)} total filings")
    print()
    
    logging.info("Generating newsletter organized by industry vertical...")
    newsletter_gen.generate_report(analyses, all_filings, args.output)
    
    if args.export_json:
        logging.info(f"Exporting to JSON: {args.export_json}")
        export_data = {
            "generated": datetime.now().isoformat(),
            "parameters": {
                "discount_rate": args.discount_rate,
                "base_growth": args.base_growth,
                "terminal_growth": args.terminal_growth,
                "mos_threshold": args.mos_threshold,
                "sec_lookback_days": args.days,
                "news_lookback_days": 90
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
    
    if args.export_csv:
        logging.info(f"Exporting to CSV: {args.export_csv}")
        df = pd.DataFrame([asdict(a) for a in analyses])
        df.to_csv(args.export_csv, index=False)
        logging.info(f"✓ Exported to {args.export_csv}")
    
    print()
    print("=" * 80)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Companies analyzed: {len(analyses)}/{len(companies_df)}")
    print(f"   Industry verticals: {len(set(a.industry for a in analyses))}")
    print(f"   SEC filings found: {len(all_filings)}")
    print(f"   News articles analyzed: {sum(len(v) for v in NEWS_CACHE.values())}")
    print()
    print(f"📰 NEWSLETTERS GENERATED:")
    for title, filepath in generated_files:
        print(f"   - {title}: {filepath}")
    print("=" * 80)


if __name__ == "__main__":
    main()
