"""
SEC EDGAR Scraper
===================================
Downloads 10-K and 10-Q filings for any public company
directly from the SEC's free public API.

HOW TO RUN:
    pip install requests
    python edgar_scraper.py

OUTPUT STRUCTURE:
    filings/
        TSLA/
            10-K/
                2021_10-K.txt
                2022_10-K.txt
                2023_10-K.txt
            10-Q/
                2023-Q1_10-Q.txt
                2023-Q2_10-Q.txt
                ...
        F/
            10-K/
                ...

ADDING MORE COMPANIES:
    Just add their ticker to the COMPANIES list at the bottom.
"""

import requests
import os
import time
import json
from datetime import datetime

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# SEC requires a User-Agent header identifying who you are.

USER_AGENT = "Hanna Boudara -- hanna.boudara@duke.edu"

# How many seconds to wait between requests.
# SEC rate limit is 10 requests/second. We stay well below that.
RATE_LIMIT_PAUSE = 0.15

# Where to save all the filings on your computer
OUTPUT_DIR = "filings"

# How far back to go
START_YEAR = 2015

# Which filing types to download
FILING_TYPES = ["10-K"]

# ── COMPANIES TO SCRAPE ────────────────────────────────────────────────────────
# Format: { "TICKER": "Company Name" }


COMPANIES = {
    "TSLA": "Tesla Inc",
    # "F":   "Ford Motor Company",      # uncomment to add
    #"NVDA": "Nvidia Corporation",     # uncomment to add
    #"LMT":  "Lockheed Martin",        # uncomment to add
    # "GM":   "General Motors",         # uncomment to add
}

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

def get_headers():
    """
    Every request to the SEC API must include a User-Agent header.
    The SEC uses this to identify who is making requests.
    Without it, your requests will be blocked.
    """
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }


def get_cik(ticker):
    """
    WHAT IS A CIK?
    The SEC doesn't use stock tickers internally — it uses CIK numbers
    (Central Index Key). Every company has a unique CIK.
    Tesla's CIK is 0001318605, for example.

    This function takes a ticker like "TSLA" and returns the CIK number
    by querying the SEC's company lookup endpoint.

    Returns the CIK as a zero-padded 10-digit string, which is the
    format the SEC API expects (e.g. "0001318605").
    """
    print(f"  Looking up CIK for {ticker}...")

    url = "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2000-01-01&enddt=2024-12-31&forms=10-K"

    # Use the company tickers JSON — the SEC provides a master list
    # of all tickers and their CIKs as a single downloadable file
    url = "https://www.sec.gov/files/company_tickers.json"

    response = requests.get(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    })

    if response.status_code != 200:
        print(f"  ERROR: Could not fetch ticker list (status {response.status_code})")
        return None

    # The response is a dict where each value has 'ticker' and 'cik_str'
    companies = response.json()

    for entry in companies.values():
        if entry["ticker"].upper() == ticker.upper():
            # CIK must be zero-padded to 10 digits for the filing API
            cik = str(entry["cik_str"]).zfill(10)
            print(f"  Found CIK: {cik}")
            return cik

    print(f"  ERROR: Ticker {ticker} not found in SEC database")
    return None


def get_filing_list(cik, filing_type):
    """
    WHAT DOES THIS DO?
    Given a CIK and filing type (e.g. "10-K"), this fetches the complete
    list of all submissions of that type from the SEC API.

    The SEC provides a submissions endpoint that returns metadata about
    every filing a company has ever made — dates, accession numbers,
    filing types, etc. We filter this down to just the type we want.

    WHAT IS AN ACCESSION NUMBER?
    Every SEC filing has a unique accession number — a string like
    "0001318605-23-000010". This is how you look up the actual document.
    Think of it as the filing's unique ID.

    Returns a list of dicts, each with:
        - accession_number: the unique filing ID
        - date: when it was filed
        - filing_type: "10-K" or "10-Q"
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    response = requests.get(url, headers=get_headers())
    time.sleep(RATE_LIMIT_PAUSE)

    if response.status_code != 200:
        print(f"  ERROR: Could not fetch filing list (status {response.status_code})")
        return []

    data = response.json()

    # The submissions JSON has a 'filings' key with a 'recent' sub-key
    # that contains parallel arrays for each field
    recent = data.get("filings", {}).get("recent", {})

    forms       = recent.get("form", [])
    dates       = recent.get("filingDate", [])
    accessions  = recent.get("accessionNumber", [])

    filings = []
    for form, date, accession in zip(forms, dates, accessions):

        # Filter to the filing type we want
        if form != filing_type:
            continue

        # Filter to our start year
        filing_year = int(date[:4])
        if filing_year < START_YEAR:
            continue

        filings.append({
            "accession_number": accession,
            "date": date,
            "filing_type": form
        })

    # Sort chronologically (oldest first)
    filings.sort(key=lambda x: x["date"])
    return filings


def get_filing_text(cik, accession_number):
    """
    WHAT DOES THIS DO?
    Given a CIK and accession number, this fetches the actual text
    content of the filing.

    HOW THE SEC FILING SYSTEM WORKS:
    Each filing is stored as a folder of files on the SEC server.
    The main document (the actual 10-K text) is usually an .htm or .txt file.
    There's also an index file that lists all documents in the filing.

    We first fetch the filing index to find the main document filename,
    then fetch that specific document.

    WHY TEXT NOT PDF?
    The SEC stores filings as HTML or plain text, not PDF. This is actually
    better for us — plain text is much easier for the NLP model to process
    than extracting text from a PDF.
    """

    # Convert accession number format: "0001318605-23-000010"
    # to folder format: "000131860523000010" (no dashes)
    accession_clean = accession_number.replace("-", "")

    # The filing index URL
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{accession_number}-index.htm"
    )

    headers_www = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    index_response = requests.get(index_url, headers=headers_www)
    time.sleep(RATE_LIMIT_PAUSE)

    if index_response.status_code != 200:
        # Try the JSON index as fallback
        json_index_url = (
            f"https://data.sec.gov/submissions/CIK{cik}.json"
        )
        return None

    # Parse the index page to find the main document
    # The main 10-K document usually has type "10-K" or "10-K/A"
    # We look for the first .htm file that is the primary document
    index_text = index_response.text
    main_doc = find_main_document(index_text, accession_clean, int(cik))

    if not main_doc:
        return None

    # Fetch the actual document
    doc_response = requests.get(main_doc, headers=headers_www)
    time.sleep(RATE_LIMIT_PAUSE)

    if doc_response.status_code != 200:
        return None

    return doc_response.text


def find_main_document(index_html, accession_clean, cik_int):
    """
    WHAT DOES THIS DO?
    Parses the filing index page to find the URL of the main document.

    The index page is an HTML table listing all files in the filing package.
    We look for the file that is the primary 10-K or 10-Q document —
    usually identified by having type "10-K", "10-Q", or being the
    largest .htm file in the package.

    This is a bit messy because the SEC filing format has changed over
    the years, so we try a few different approaches.
    """
    import re

    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/"

    # Approach 1: look for the document explicitly tagged as 10-K type
    pattern = r'<td[^>]*>(?:10-K|10-K/A)</td>.*?<a href="([^"]+\.htm)"'
    matches = re.findall(pattern, index_html, re.IGNORECASE | re.DOTALL)
    if matches:
        # Filter out XBRL/iXBRL files which start with R or are in xbrl folders
        for match in matches:
            fname = match.lower()
            if not any(x in fname for x in ['r2.htm', 'r3.htm', 'xbrl', 'inline']):
                filename = match.split("/")[-1]
                return base_url + filename

    # Approach 2: look for files with 10k or annual in the name
    pattern2 = r'<a href="([^"]*(?:10k|10-k|annual|form10)[^"]*\.htm)"'
    matches2 = re.findall(pattern2, index_html, re.IGNORECASE)
    if matches2:
        filename = matches2[0].split("/")[-1]
        return base_url + filename

    # Approach 3: find the LARGEST .htm file — that's almost always the main doc
    pattern3 = r'<td[^>]*>(\d+)</td>\s*<td[^>]*><a href="([^"]+\.htm)"'
    size_matches = re.findall(pattern3, index_html, re.IGNORECASE)
    if size_matches:
        # Sort by file size descending, take the biggest
        size_matches.sort(key=lambda x: int(x[0]), reverse=True)
        for size, match in size_matches:
            fname = match.lower()
            if not any(x in fname for x in ['exhibit', 'ex-', 'xsd', 'css', 'xbrl']):
                filename = match.split("/")[-1]
                return base_url + filename

    # Approach 4: fallback to first .htm that isn't an exhibit
    pattern4 = r'href="([^"]+\.htm)"'
    matches4 = re.findall(pattern4, index_html, re.IGNORECASE)
    for match in matches4:
        fname = match.lower()
        if not any(x in fname for x in ['exhibit', 'ex-', 'stylesheet', 'css', 'xsd', 'xbrl']):
            filename = match.split("/")[-1]
            return base_url + filename

    return None


def clean_text(html_text):
    """
    WHAT DOES THIS DO?
    The raw filing text from EDGAR is HTML — full of tags, inline CSS,
    JavaScript, and other noise. We strip all of that to get clean
    plain text that the NLP model can actually process.

    We keep paragraph breaks so the document structure is preserved —
    this matters for the NLP extraction later because you want to know
    whether a sentence appears in the Risk Factors section vs. the
    Financial Statements section.
    """
    
    import re

    # Remove XBRL/iXBRL tags but keep the text inside them
    text = re.sub(r'<ix:[^>]+>', ' ', html_text, flags=re.IGNORECASE)
    text = re.sub(r'</ix:[^>]+>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<xbrl[^>]*>.*?</xbrl>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove script and style blocks entirely
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    # Replace paragraph/div/br tags with newlines
    text = re.sub(r'<(?:p|div|br|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&#160;', ' ')
    text = text.replace('&ldquo;', '"')
    text = text.replace('&rdquo;', '"')
    text = text.replace('&rsquo;', "'")

    # Collapse multiple spaces and blank lines
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def save_filing(ticker, filing_type, date, text):
    """
    Saves a cleaned filing to disk in an organized folder structure.

    For 10-Ks: filings/TSLA/10-K/2022_10-K.txt
    For 10-Qs: filings/TSLA/10-Q/2022-Q2_10-Q.txt
    """
    year = date[:4]
    month = int(date[5:7])

    if filing_type == "10-Q":
        # Determine quarter from filing month
        # Q1 filed ~May, Q2 ~Aug, Q3 ~Nov, Q4/annual = 10-K
        if month <= 5:
            quarter = "Q1"
        elif month <= 8:
            quarter = "Q2"
        else:
            quarter = "Q3"
        filename = f"{year}-{quarter}_10-Q.txt"
    else:
        filename = f"{year}_10-K.txt"

    folder = os.path.join(OUTPUT_DIR, ticker, filing_type)
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


def scrape_company(ticker, company_name):
    """
    MAIN FUNCTION PER COMPANY.
    Orchestrates the full scrape for one company:
    1. Look up CIK
    2. For each filing type (10-K, 10-Q):
       a. Get list of all filings
       b. For each filing, download and clean the text
       c. Save to disk
    """
    print(f"\n{'='*60}")
    print(f"Scraping: {company_name} ({ticker})")
    print(f"{'='*60}")

    # Step 1: get CIK
    cik = get_cik(ticker)
    if not cik:
        print(f"  FAILED: Could not find CIK for {ticker}")
        return

    summary = {}

    for filing_type in FILING_TYPES:
        print(f"\n  Filing type: {filing_type}")

        # Step 2: get list of filings
        filings = get_filing_list(cik, filing_type)
        print(f"  Found {len(filings)} {filing_type} filings since {START_YEAR}")

        saved = 0
        failed = 0

        for filing in filings:
            date      = filing["date"]
            accession = filing["accession_number"]

            # Check if already downloaded (avoid re-downloading)
            year  = date[:4]
            month = int(date[5:7])
            if filing_type == "10-Q":
                quarter  = "Q1" if month <= 5 else "Q2" if month <= 8 else "Q3"
                filename = f"{year}-{quarter}_10-Q.txt"
            else:
                filename = f"{year}_10-K.txt"

            filepath = os.path.join(OUTPUT_DIR, ticker, filing_type, filename)
            if os.path.exists(filepath):
                print(f"    SKIP (already exists): {filename}")
                saved += 1
                continue

            print(f"    Downloading {filing_type} filed {date}...", end=" ")

            # Step 3: download
            raw_text = get_filing_text(cik, accession)

            if not raw_text:
                print("FAILED (no text retrieved)")
                failed += 1
                continue

            # Step 4: clean
            clean = clean_text(raw_text)

            if len(clean) < 10000:
                # Suspiciously short — probably got an index page, not the filing
                print(f"FAILED (text too short: {len(clean)} chars)")
                failed += 1
                continue

            # Step 5: save
            saved_path = save_filing(ticker, filing_type, date, clean)
            print(f"OK ({len(clean):,} chars → {os.path.basename(saved_path)})")
            saved += 1

        summary[filing_type] = {"saved": saved, "failed": failed}

    # Print summary
    print(f"\n  Summary for {ticker}:")
    for ft, counts in summary.items():
        print(f"    {ft}: {counts['saved']} saved, {counts['failed']} failed")


def print_corpus_stats():
    """
    After scraping, print a summary of what's in the corpus.
    Shows how many filings per company per type, and total word counts.
    """
    print(f"\n{'='*60}")
    print("CORPUS SUMMARY")
    print(f"{'='*60}")

    total_files = 0
    total_chars = 0

    for ticker in sorted(os.listdir(OUTPUT_DIR)):
        ticker_dir = os.path.join(OUTPUT_DIR, ticker)
        if not os.path.isdir(ticker_dir):
            continue

        print(f"\n  {ticker}")
        for filing_type in FILING_TYPES:
            type_dir = os.path.join(ticker_dir, filing_type)
            if not os.path.isdir(type_dir):
                continue

            files = sorted(os.listdir(type_dir))
            if not files:
                continue

            type_chars = 0
            for fname in files:
                fpath = os.path.join(type_dir, fname)
                size = os.path.getsize(fpath)
                type_chars += size

            total_files  += len(files)
            total_chars  += type_chars
            avg_kb = (type_chars / len(files)) / 1024

            print(f"    {filing_type}: {len(files)} filings | avg {avg_kb:.0f} KB each")
            print(f"      Files: {', '.join(files)}")

    print(f"\n  TOTAL: {total_files} filings | {total_chars/1e6:.1f} MB")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("SEC EDGAR Scraper")
    print("="*60)
    print(f"Companies : {list(COMPANIES.keys())}")
    print(f"Filing types: {FILING_TYPES}")
    print(f"Start year  : {START_YEAR}")
    print(f"Output dir  : {os.path.abspath(OUTPUT_DIR)}")
    print()

    # Warn if User-Agent is still the placeholder
    if "YourName" in USER_AGENT:
        print("WARNING: Please update USER_AGENT at the top of the script")
        print("         with your actual name and email before running.")
        print("         The SEC requires this.")
        print()

    # Scrape each company
    for ticker, name in COMPANIES.items():
        scrape_company(ticker, name)

    # Print final corpus stats
    print_corpus_stats()

    print(f"\nDone. Filings saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Next step: run nlp_extractor.py on this corpus")
