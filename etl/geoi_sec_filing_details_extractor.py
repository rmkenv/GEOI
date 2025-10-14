"""
GEOI SEC Filing Details Extractor - IPO & Insider Trading Analysis

ETL Part 2

This script processes SEC filing data to extract detailed information about:
1. IPO-related filings (S-1, S-1/A, F-1, F-1/A) - prospectus details, offering prices, underwriters
2. Insider trading (Forms 3, 4, 5) - ownership transactions, reporting owners, securities traded

The script downloads and parses the actual filing documents from SEC EDGAR:
- For IPOs: Downloads primary prospectus documents and extracts key offering details
- For Insider Trading: Downloads ownership XML files and parses structured transaction data

Input: Parquet file from GEOI SEC filings ETL (with ticker, CIK, formType, accessionNo)
Output: 
  - ipo_signals.csv: IPO offering details with saved prospectus text files
  - insider_transactions.csv: Parsed insider transactions with owner and security details
  - filing_details/: Directory with downloaded filing documents

API: SEC EDGAR Archives (requires User-Agent header, respects rate limits)

Usage:
    1. Update USER_AGENT with your contact information
    2. Run after geoi_sec_filings_etl.py has generated filing data
    3. Call: ipo_df, insider_df = run_from_parquet("path/to/filings.parquet")

Author: RMK
Date: 2025-10-14
Filename: geoi_sec_filing_details_extractor.py
"""

import os
import re
import time
import json
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

USER_AGENT = "Your Name Your Company you@example.com"  # ⚠️ CHANGE THIS TO YOUR INFO
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY = 0.4  # seconds between requests - be respectful to SEC servers

OUTPUT_BASE = Path("./geospatial_finance_data")
DETAILS_DIR = OUTPUT_BASE / "filing_details"
DETAILS_DIR.mkdir(parents=True, exist_ok=True)

IPO_FORMS = {"S-1", "S-1/A", "F-1", "F-1/A"}
INSIDER_FORMS = {"3", "4", "5"}

def cik_no_leading_zeros(cik: str) -> str:
    """Remove leading zeros from CIK for URL construction"""
    return str(int(str(cik)))

def accession_nodash(accession: str) -> str:
    """Remove dashes from accession number for URL construction"""
    return accession.replace("-", "")

def filing_index_json_url(cik: str, accession: str) -> str:
    """Construct URL for SEC filing index.json"""
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros(cik)}/{accession_nodash(accession)}/index.json"

def get_filing_index(cik: str, accession: str) -> Optional[dict]:
    """Fetch filing index.json which lists all documents in a filing"""
    url = filing_index_json_url(cik, accession)
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()
    except Exception as e:
        print(f"Error fetching index for {cik}/{accession}: {e}")
        return None

def build_doc_url(cik: str, accession: str, filename: str) -> str:
    """Construct URL for a specific document within a filing"""
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros(cik)}/{accession_nodash(accession)}/{filename}"

def safe_filename(*parts) -> str:
    """Create a safe filename from parts"""
    base = "_".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", base)[:240]

def download_text(url: str) -> Optional[str]:
    """Download text content from URL"""
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        time.sleep(REQUEST_DELAY)
        try:
            return r.text
        except Exception:
            return r.content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def get_xml_text(element, default=""):
    """Safely extract text from XML element"""
    if element is not None and element.text:
        return element.text.strip()
    return default

def parse_insider_xml_robust(xml_text: str) -> list[dict]:
    """
    Robust XML parser for SEC Forms 3, 4, 5 (ownership documents)
    Extracts: issuer info, reporting owners, non-derivative transactions, derivative transactions
    """
    transactions = []
    
    try:
        # Parse XML
        root = ET.fromstring(xml_text)
        
        # Handle namespace if present
        ns = {'': ''}
        if root.tag.startswith('{'):
            ns_match = re.match(r'\{(.*?)\}', root.tag)
            if ns_match:
                ns = {'ns': ns_match.group(1)}
        
        # Helper to find with optional namespace
        def find_elem(parent, path):
            try:
                return parent.find(path) or parent.find(f"ns:{path}", ns) or parent.find(f".//{path}") or parent.find(f".//ns:{path}", ns)
            except:
                return None
        
        def findall_elem(parent, path):
            try:
                result = parent.findall(path) or parent.findall(f"ns:{path}", ns) or parent.findall(f".//{path}") or parent.findall(f".//ns:{path}", ns)
                return result if result else []
            except:
                return []
        
        # Extract issuer information
        issuer = find_elem(root, "issuer")
        issuer_cik = get_xml_text(find_elem(issuer, "issuerCik")) if issuer else ""
        issuer_name = get_xml_text(find_elem(issuer, "issuerName")) if issuer else ""
        issuer_symbol = get_xml_text(find_elem(issuer, "issuerTradingSymbol")) if issuer else ""
        
        # Extract reporting owner(s) information
        reporting_owners = []
        for owner in findall_elem(root, "reportingOwner"):
            owner_id = find_elem(owner, "reportingOwnerId")
            owner_name = get_xml_text(find_elem(owner_id, "rptOwnerName")) if owner_id else ""
            owner_cik = get_xml_text(find_elem(owner_id, "rptOwnerCik")) if owner_id else ""
            
            # Get relationship info
            relationship = find_elem(owner, "reportingOwnerRelationship")
            is_director = get_xml_text(find_elem(relationship, "isDirector"), "0") if relationship else "0"
            is_officer = get_xml_text(find_elem(relationship, "isOfficer"), "0") if relationship else "0"
            is_ten_percent = get_xml_text(find_elem(relationship, "isTenPercentOwner"), "0") if relationship else "0"
            is_other = get_xml_text(find_elem(relationship, "isOther"), "0") if relationship else "0"
            officer_title = get_xml_text(find_elem(relationship, "officerTitle"), "") if relationship else ""
            
            # Get address
            address = find_elem(owner, "reportingOwnerAddress")
            street1 = get_xml_text(find_elem(address, "rptOwnerStreet1")) if address else ""
            street2 = get_xml_text(find_elem(address, "rptOwnerStreet2")) if address else ""
            city = get_xml_text(find_elem(address, "rptOwnerCity")) if address else ""
            state = get_xml_text(find_elem(address, "rptOwnerState")) if address else ""
            zipcode = get_xml_text(find_elem(address, "rptOwnerZipCode")) if address else ""
            
            reporting_owners.append({
                "name": owner_name,
                "cik": owner_cik,
                "isDirector": is_director,
                "isOfficer": is_officer,
                "isTenPercentOwner": is_ten_percent,
                "isOther": is_other,
                "officerTitle": officer_title,
                "address": f"{street1} {street2} {city}, {state} {zipcode}".strip()
            })
        
        owners_summary = "; ".join([f"{o['name']} ({o['officerTitle'] or 'N/A'})" for o in reporting_owners])
        
        # Parse non-derivative transactions
        for ndt in findall_elem(root, "nonDerivativeTransaction"):
            security_title = get_xml_text(find_elem(ndt, "securityTitle"))
            
            trans_coding = find_elem(ndt, "transactionCoding")
            trans_form_type = get_xml_text(find_elem(trans_coding, "transactionFormType")) if trans_coding else ""
            trans_code = get_xml_text(find_elem(trans_coding, "transactionCode")) if trans_coding else ""
            equity_swap = get_xml_text(find_elem(trans_coding, "equitySwapInvolved")) if trans_coding else ""
            
            trans_date = get_xml_text(find_elem(ndt, "transactionDate"))
            
            trans_amounts = find_elem(ndt, "transactionAmounts")
            shares = get_xml_text(find_elem(trans_amounts, "transactionShares")) if trans_amounts else ""
            price = get_xml_text(find_elem(trans_amounts, "transactionPricePerShare")) if trans_amounts else ""
            acquired_disposed = get_xml_text(find_elem(trans_amounts, "transactionAcquiredDisposedCode")) if trans_amounts else ""
            
            post_trans = find_elem(ndt, "postTransactionAmounts")
            shares_owned = get_xml_text(find_elem(post_trans, "sharesOwnedFollowingTransaction")) if post_trans else ""
            
            ownership = find_elem(ndt, "ownershipNature")
            direct_indirect = get_xml_text(find_elem(ownership, "directOrIndirectOwnership")) if ownership else ""
            
            # Footnotes
            footnote_ids = []
            for fn_ref in findall_elem(ndt, "footnoteId"):
                footnote_ids.append(get_xml_text(fn_ref))
            
            transactions.append({
                "isDerivative": False,
                "securityTitle": security_title,
                "transactionDate": trans_date,
                "transactionCode": trans_code,
                "transactionFormType": trans_form_type,
                "equitySwapInvolved": equity_swap,
                "shares": shares,
                "pricePerShare": price,
                "acquiredDisposed": acquired_disposed,
                "sharesOwnedFollowing": shares_owned,
                "directOrIndirect": direct_indirect,
                "footnoteIds": ", ".join(footnote_ids),
                "issuerCIK": issuer_cik,
                "issuerName": issuer_name,
                "issuerSymbol": issuer_symbol,
                "reportingOwners": owners_summary,
                "reportingOwnerDetails": str(reporting_owners)
            })
        
        # Parse derivative transactions
        for dt in findall_elem(root, "derivativeTransaction"):
            security_title = get_xml_text(find_elem(dt, "securityTitle"))
            conversion_price = get_xml_text(find_elem(dt, "conversionOrExercisePrice"))
            
            trans_coding = find_elem(dt, "transactionCoding")
            trans_form_type = get_xml_text(find_elem(trans_coding, "transactionFormType")) if trans_coding else ""
            trans_code = get_xml_text(find_elem(trans_coding, "transactionCode")) if trans_coding else ""
            equity_swap = get_xml_text(find_elem(trans_coding, "equitySwapInvolved")) if trans_coding else ""
            
            trans_date = get_xml_text(find_elem(dt, "transactionDate"))
            
            trans_amounts = find_elem(dt, "transactionAmounts")
            shares = get_xml_text(find_elem(trans_amounts, "transactionShares")) if trans_amounts else ""
            price = get_xml_text(find_elem(trans_amounts, "transactionPricePerShare")) if trans_amounts else ""
            acquired_disposed = get_xml_text(find_elem(trans_amounts, "transactionAcquiredDisposedCode")) if trans_amounts else ""
            
            exercise_date = get_xml_text(find_elem(dt, "exerciseDate"))
            expiration_date = get_xml_text(find_elem(dt, "expirationDate"))
            
            underlying = find_elem(dt, "underlyingSecurity")
            underlying_title = get_xml_text(find_elem(underlying, "underlyingSecurityTitle")) if underlying else ""
            underlying_shares = get_xml_text(find_elem(underlying, "underlyingSecurityShares")) if underlying else ""
            
            post_trans = find_elem(dt, "postTransactionAmounts")
            shares_owned = get_xml_text(find_elem(post_trans, "sharesOwnedFollowingTransaction")) if post_trans else ""
            
            ownership = find_elem(dt, "ownershipNature")
            direct_indirect = get_xml_text(find_elem(ownership, "directOrIndirectOwnership")) if ownership else ""
            
            # Footnotes
            footnote_ids = []
            for fn_ref in findall_elem(dt, "footnoteId"):
                footnote_ids.append(get_xml_text(fn_ref))
            
            transactions.append({
                "isDerivative": True,
                "securityTitle": security_title,
                "conversionOrExercisePrice": conversion_price,
                "transactionDate": trans_date,
                "transactionCode": trans_code,
                "transactionFormType": trans_form_type,
                "equitySwapInvolved": equity_swap,
                "shares": shares,
                "pricePerShare": price,
                "acquiredDisposed": acquired_disposed,
                "exerciseDate": exercise_date,
                "expirationDate": expiration_date,
                "underlyingSecurityTitle": underlying_title,
                "underlyingSecurityShares": underlying_shares,
                "sharesOwnedFollowing": shares_owned,
                "directOrIndirect": direct_indirect,
                "footnoteIds": ", ".join(footnote_ids),
                "issuerCIK": issuer_cik,
                "issuerName": issuer_name,
                "issuerSymbol": issuer_symbol,
                "reportingOwners": owners_summary,
                "reportingOwnerDetails": str(reporting_owners)
            })
        
        # If no transactions found, still return issuer/owner info
        if not transactions and reporting_owners:
            transactions.append({
                "isDerivative": None,
                "securityTitle": "",
                "transactionDate": "",
                "transactionCode": "",
                "shares": "",
                "pricePerShare": "",
                "issuerCIK": issuer_cik,
                "issuerName": issuer_name,
                "issuerSymbol": issuer_symbol,
                "reportingOwners": owners_summary,
                "reportingOwnerDetails": str(reporting_owners)
            })
    
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return []
    except Exception as e:
        print(f"Error parsing insider XML: {e}")
        return []
    
    return transactions

def extract_ipo_signals(text: str) -> dict:
    """
    Extract IPO-related information from S-1/F-1 prospectus text
    Uses regex patterns to find common IPO details
    """
    res = {}
    
    # Proposed maximum aggregate offering price
    m = re.search(r"proposed maximum aggregate offering price[^$]*\$([\d,\.]+)", text, flags=re.I)
    if m: 
        res["proposedMaxAggregateOfferingPrice"] = m.group(1)
    
    # Price range per share
    m = re.search(r"price.*?per share.*?\$([\d,\.]+).*?to.*?\$([\d,\.]+)", text, flags=re.I)
    if m:
        res["priceRangeLow"] = m.group(1)
        res["priceRangeHigh"] = m.group(2)
    
    # Underwriters
    m = re.search(r"underwriter[s]?:?\s*(.*?)(?:\n|\.)", text, flags=re.I)
    if m: 
        res["underwriters"] = m.group(1).strip()
    
    # Intended exchange
    m = re.search(r"(?:we intend to list|will be listed|listing on).*?(?:on|the)\s+([A-Za-z\-\s&]+?)(?:\.|under)", text, flags=re.I)
    if m: 
        res["intendedExchange"] = m.group(1).strip()
    
    # Ticker symbol
    m = re.search(r"ticker symbol\s*(?:will be|is|:)\s*[\"']?([A-Z\.]{1,10})[\"']?", text, flags=re.I)
    if m: 
        res["intendedTicker"] = m.group(1).strip(".")
    
    # Number of shares offered
    m = re.search(r"offering.*?([\d,]+)\s+shares", text, flags=re.I)
    if m:
        res["sharesOffered"] = m.group(1)
    
    # Use of proceeds
    m = re.search(r"use of proceeds[:\s]+(.*?)(?:\n\n|RISK FACTORS)", text, flags=re.I | re.DOTALL)
    if m:
        res["useOfProceeds"] = m.group(1).strip()[:500]  # Limit to 500 chars
    
    return res

def process_filings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process filings DataFrame to extract IPO and insider trading details
    
    Input df columns required: ['ticker', 'cik', 'formType', 'accessionNo']
    Returns:
      ipo_df: extracted IPO signals with raw text snapshot path
      insider_df: parsed insider transactions with full details
    """
    ipo_rows = []
    insider_rows = []

    total = len(df)
    for i, row in df.iterrows():
        if i % 10 == 0:
            print(f"Processing {i+1}/{total}...")
        
        form = str(row.get("formType", "")).strip()
        cik = str(row.get("cik", "")).zfill(10)
        accession = str(row.get("accessionNo", "")).strip()
        ticker = row.get("ticker")

        if not cik or not accession:
            continue

        idx = get_filing_index(cik, accession)
        if not idx or "directory" not in idx or "item" not in idx["directory"]:
            continue

        files = idx["directory"]["item"]

        # IPO filings: look for S-1/F-1 primary doc (.htm/.html/.txt)
        if form in IPO_FORMS:
            primary = None
            for f in files:
                name = f.get("name", "")
                if name.lower().endswith((".htm", ".html", ".txt")):
                    primary = name
                    break
            
            if primary:
                url = build_doc_url(cik, accession, primary)
                text = download_text(url)
                if text:
                    out_path = DETAILS_DIR / safe_filename(f"{ticker}_{form}_{accession}_prospectus.txt")
                    out_path.write_text(text, encoding="utf-8", errors="ignore")
                    signals = extract_ipo_signals(text)
                    ipo_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "formType": form,
                        "accessionNo": accession,
                        "primaryDocUrl": url,
                        "savedText": str(out_path),
                        **signals
                    })

        # Insider filings: 3/4/5 XML ownership docs
        if form in INSIDER_FORMS:
            xml_candidate = None
            # Look for ownership XML files
            for f in files:
                name = f.get("name", "")
                if name.lower().endswith(".xml") and any(x in name.lower() for x in ["ownership", "form", "f345", "xslF345"]):
                    xml_candidate = name
                    break
            
            # Fallback: pick any XML if none matched
            if not xml_candidate:
                for f in files:
                    name = f.get("name", "")
                    if name.lower().endswith(".xml"):
                        xml_candidate = name
                        break
            
            if xml_candidate:
                url = build_doc_url(cik, accession, xml_candidate)
                xml_text = download_text(url)
                if xml_text:
                    out_path = DETAILS_DIR / safe_filename(f"{ticker}_{form}_{accession}_ownership.xml")
                    out_path.write_text(xml_text, encoding="utf-8", errors="ignore")
                    txs = parse_insider_xml_robust(xml_text)
                    for t in txs:
                        insider_rows.append({
                            "ticker": ticker,
                            "cik": cik,
                            "formType": form,
                            "accessionNo": accession,
                            "docUrl": url,
                            "savedXml": str(out_path),
                            **t
                        })

    ipo_df = pd.DataFrame(ipo_rows)
    insider_df = pd.DataFrame(insider_rows)
    return ipo_df, insider_df

def run_from_parquet(parquet_path: str, last_n_years: int = 5):
    """
    Main function to process SEC filings from parquet file
    
    Args:
        parquet_path: Path to parquet file from GEOI SEC filings ETL
        last_n_years: Number of years to look back (default: 5)
    
    Returns:
        tuple: (ipo_df, insider_df) DataFrames with extracted data
    """
    print(f"Loading filings from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Focus on last N years
    df["filedAt"] = pd.to_datetime(df["filedAt"], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=last_n_years)
    df = df[df["filedAt"] >= cutoff]

    # Keep needed columns
    needed_cols = ["ticker", "cik", "formType", "accessionNo"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Limit to target forms
    df_target = df[df["formType"].isin(IPO_FORMS.union(INSIDER_FORMS))].copy()
    print(f"\nProcessing {len(df_target)} filings (IPO + Insider) from last {last_n_years} years...")
    print(f"  IPO forms (S-1, F-1): {len(df_target[df_target['formType'].isin(IPO_FORMS)])}")
    print(f"  Insider forms (3, 4, 5): {len(df_target[df_target['formType'].isin(INSIDER_FORMS)])}")

    ipo_df, insider_df = process_filings(df_target)

    # Save outputs
    ipo_out = OUTPUT_BASE / "ipo_signals.csv"
    insider_out = OUTPUT_BASE / "insider_transactions.csv"
    
    ipo_df.to_csv(ipo_out, index=False)
    insider_df.to_csv(insider_out, index=False)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"\n📊 IPO Signals:")
    print(f"  Total IPO filings processed: {len(ipo_df)}")
    print(f"  Saved to: {ipo_out}")
    if len(ipo_df) > 0:
        print(f"  Columns: {', '.join(ipo_df.columns)}")
    
    print(f"\n📊 Insider Transactions:")
    print(f"  Total transactions extracted: {len(insider_df)}")
    print(f"  Saved to: {insider_out}")
    if len(insider_df) > 0:
        print(f"  Columns: {', '.join(insider_df.columns)}")
    
    print(f"\n📁 Filing documents saved to: {DETAILS_DIR}")
    
    return ipo_df, insider_df

if __name__ == "__main__":
    # Example usage - update path to your filings parquet
    ipo_df, insider_df = run_from_parquet(
        "./geospatial_finance_data/sec_filings_last5y_20251014_150823.parquet",
        last_n_years=5
    )
