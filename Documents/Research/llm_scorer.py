"""
LLM Scorer for Rare Earth Disclosure
=====================================
Reads the .txt filings already downloaded by edgar_scraper.py
and scores each one for rare earth disclosure quality.

HOW TO RUN:
    pip install anthropic python-dotenv pandas
    python llm_scorer.py

OUTPUT:
    materiality.csv   — which companies are exposed to rare earths
    dqi_scores.csv    — detailed disclosure quality scores
"""

import os
import json
import time
import re
import anthropic
import pandas as pd
from dotenv import load_dotenv

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

load_dotenv()  # reads your .env file automatically

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

FILINGS_DIR = "filings"   # same folder your edgar scraper uses
MODEL = "claude-haiku-4-5-20251001"  # cheap + fast, perfect for extraction

# Which tickers to score — must already be downloaded by edgar_scraper.py
TICKERS = [
    "TSLA",
    # "NVDA",
    # "AAPL",
    #"LMT"
    # add more once you've tested Tesla works
]

# ── SECTION EXTRACTOR ─────────────────────────────────────────────────────────

def extract_sections(text):
    """
    Pulls Item 1 (Business) and Item 1A (Risk Factors)
    More flexible regex to handle different formatting styles
    """

    risk_text = ""
    business_text = ""

    # Try multiple patterns for Item 1A — Risk Factors
    risk_patterns = [
        r'item\s+1a[\.\s\:]+risk factors(.*?)item\s+1b',
        r'item\s+1a[\.\s\:]+risk factors(.*?)item\s+2',
        r'risk factors(.*?)unresolved comments',
        r'risk factors(.*?)item\s+2',
    ]

    for pattern in risk_patterns:
        risk_match = re.search(pattern, text.lower(), re.DOTALL)
        if risk_match:
            start = risk_match.start(1)
            end = risk_match.end(1)
            risk_text = text[start:end][:8000]
            break

    # Try multiple patterns for Item 1 — Business
    business_patterns = [
        r'item\s+1[\.\s\:]+business(.*?)item\s+1a',
        r'item\s+1[\.\s\:]+business(.*?)risk factors',
        r'item\s+1[\.\s\:]+business(.*?)item\s+2',
    ]

    for pattern in business_patterns:
        business_match = re.search(pattern, text.lower(), re.DOTALL)
        if business_match:
            start = business_match.start(1)
            end = business_match.end(1)
            business_text = text[start:end][:4000]
            break

    return risk_text, business_text

# ── STAGE 1: MATERIALITY CHECK ────────────────────────────────────────────────

def assess_materiality(ticker, year, business_text):
    """
    Before scoring disclosure quality, check whether this company
    should even be expected to disclose rare earth exposure.
    A restaurant scoring 0 doesn't mean bad disclosure — it means
    rare earths simply aren't relevant to their business.
    """

    if not business_text:
        return {
            "ticker": ticker, "year": year,
            "materiality": "none", "include_in_study": False,
            "reasoning": "No business text found", "sector": "unknown"
        }

    prompt = f"""
You are an expert in industrial supply chains and critical minerals.

Read this business description from {ticker}'s {year} 10-K:

{business_text}

Assess whether this company likely has material exposure to rare earth 
metals or critical minerals based purely on what they make or do.

HIGH exposure: EV makers, battery manufacturers, consumer electronics, 
defense, wind/solar, semiconductors, industrial motors, MRI machines
MEDIUM exposure: aerospace, robotics, medical devices, optical equipment  
LOW/NONE: restaurants, banks, pure software, retail, healthcare services

Return ONLY valid JSON, no other text, no explanation:
{{
    "materiality": "high" or "medium" or "low" or "none",
    "reasoning": "1-2 sentence explanation",
    "sector": "sector name",
    "include_in_study": true or false
}}

Set include_in_study to true only for high or medium.
"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text)
        result["ticker"] = ticker
        result["year"] = year
        return result

    except Exception as e:
        print(f"    ⚠️  Materiality check failed: {e}")
        return {
            "ticker": ticker, "year": year,
            "materiality": "unknown", "include_in_study": False,
            "reasoning": f"Error: {e}", "sector": "unknown"
        }


# ── STAGE 2: DETAILED DQI SCORING ────────────────────────────────────────────

def score_filing(ticker, year, risk_text, business_text):
    """
    Scores the filing across 5 layers:
      Layer 1 — Which minerals are mentioned, how often
      Layer 2 — Supply chain structure (direct buyer? suppliers named?)
      Layer 3 — Quantitative exposure ($ amounts, % of COGS)
      Layer 4 — Risk awareness (geopolitical, regulatory, price volatility)
      Layer 5 — Mitigation (hedging, diversification, R&D substitutes)
    
    Each layer scored 0-20. Weighted DQI out of 100.
    """

    prompt = f"""
You are a financial analyst specializing in critical minerals and supply chain risk.

Analyze these excerpts from {ticker}'s {year} 10-K filing.

RISK FACTORS:
{risk_text}

BUSINESS DESCRIPTION:
{business_text}

Return ONLY a valid JSON object, no other text, no markdown backticks.

{{
    "layer1_mineral_identification": {{
        "specific_minerals_mentioned": [
            {{"name": "mineral name", "mention_count": 0, "context": "brief context"}}
        ],
        "generic_categories_mentioned": ["e.g. rare earth metals, critical minerals"],
        "total_distinct_minerals": 0,
        "mentioned_in_dedicated_risk_section": true,
        "score": 0
    }},
    "layer2_supply_chain": {{
        "buys_directly_from_miners": "true or false or unclear",
        "dependent_on_suppliers": "true or false or unclear",
        "suppliers_named": [{{"name": "supplier name", "country": "country if mentioned"}}],
        "china_dependency_mentioned": false,
        "single_source_risk_mentioned": false,
        "long_term_agreements_mentioned": false,
        "score": 0
    }},
    "layer3_quantitative_exposure": {{
        "dollar_value_mentioned": false,
        "dollar_value": null,
        "percentage_of_cogs_mentioned": false,
        "percentage_of_cogs": null,
        "volume_figures_mentioned": false,
        "price_sensitivity_mentioned": false,
        "price_sensitivity_detail": null,
        "inventory_buffer_mentioned": false,
        "score": 0
    }},
    "layer4_risk_awareness": {{
        "price_volatility_mentioned": false,
        "geopolitical_risk_mentioned": false,
        "geopolitical_detail": null,
        "regulatory_risk_mentioned": false,
        "regulations_named": [],
        "substitution_risk_mentioned": false,
        "esg_sourcing_mentioned": false,
        "forward_looking_language": false,
        "score": 0
    }},
    "layer5_mitigation": {{
        "geographic_diversification_mentioned": false,
        "supplier_diversification_mentioned": false,
        "financial_hedging_mentioned": false,
        "hedging_instruments_named": [],
        "rd_substitutes_mentioned": false,
        "recycling_mentioned": false,
        "stockpiling_mentioned": false,
        "vertical_integration_mentioned": false,
        "score": 0
    }},
    "weighted_dqi": 0,
    "qualitative_summary": "3-4 sentence overall assessment of disclosure quality",
    "most_informative_excerpt": "the single most useful quote from the filing about mineral exposure"
}}

Score each layer 0-20 based on detail and specificity.
Compute weighted_dqi as: (l1*0.20 + l2*0.20 + l3*0.25 + l4*0.20 + l5*0.15) * 5
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text

    # sometimes the model wraps in ```json ... ``` so strip that
    raw = re.sub(r'^```json\s*', '', raw.strip())
    raw = re.sub(r'```$', '', raw.strip())

    result = json.loads(raw)
    result["ticker"] = ticker
    result["year"] = year
    return result


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(tickers):

    all_materiality = []
    all_scores = []

    for ticker in tickers:
        folder = os.path.join(FILINGS_DIR, ticker, "10-K")

        if not os.path.exists(folder):
            print(f"\n❌ No filings found for {ticker}")
            print(f"   Run edgar_scraper.py first with {ticker} in COMPANIES")
            continue

        files = sorted(os.listdir(folder))
        print(f"\n{'='*55}")
        print(f"Processing {ticker} — {len(files)} filings found")
        print(f"{'='*55}")

        for filename in files:
            if not filename.endswith(".txt"):
                continue

            year = filename.split("_")[0]  # e.g. "2023" from "2023_10-K.txt"
            filepath = os.path.join(folder, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            print(f"\n  📄 {ticker} {year}")

            # Extract sections
            risk_text, business_text = extract_sections(text)

            if not risk_text and not business_text:
                print(f"  ⚠️  Could not extract sections — skipping")
                continue

            print(f"  Extracted: {len(risk_text)} chars risk | {len(business_text)} chars business")

            # ── Stage 1: materiality ──────────────────────────────────
            mat = assess_materiality(ticker, year, business_text)
            all_materiality.append(mat)

            icon = "✅" if mat["include_in_study"] else "⏭️"
            print(f"  {icon} Materiality: {mat['materiality']} — {mat['reasoning']}")

            # ── Stage 2: score only if material ──────────────────────
            if mat["include_in_study"]:
                try:
                    score = score_filing(ticker, year, risk_text, business_text)
                    score["sector"] = mat["sector"]
                    score["materiality"] = mat["materiality"]
                    all_scores.append(score)
                    print(f"  📊 DQI: {score['weighted_dqi']:.1f}/100")
                    print(f"  💬 {score['qualitative_summary'][:120]}...")

                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parse error: {e}")
                except Exception as e:
                    print(f"  ❌ Scoring failed: {e}")

            time.sleep(0.5)  # be nice to the API

    # ── Save results ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Saving results...")

    pd.DataFrame(all_materiality).to_csv("materiality.csv", index=False)
    print(f"✅ materiality.csv — {len(all_materiality)} rows")

    if all_scores:
        pd.DataFrame(all_scores).to_csv("dqi_scores.csv", index=False)
        print(f"✅ dqi_scores.csv — {len(all_scores)} rows")
    else:
        print("⚠️  No scores to save — check materiality filter")

    print(f"\nDone! Open materiality.csv and dqi_scores.csv to see results.")
    return all_materiality, all_scores


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("Rare Earth Disclosure Scorer")
    print("="*55)
    print(f"Tickers  : {TICKERS}")
    print(f"Model    : {MODEL}")
    print(f"Filings  : {os.path.abspath(FILINGS_DIR)}")
    print()
    run_pipeline(TICKERS)
    