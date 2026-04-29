import pandas as pd

# Load the S&P 500 list
df = pd.read_csv('sp500_companies.csv')

# ── TIER 1: HIGH EXPECTED RARE EARTH EXPOSURE ─────────────────────────────────
high_exposure_subindustries = [
    # Semiconductors
    'Semiconductors',
    'Semiconductor Materials & Equipment',
    'Electronic Components',
    'Electronic Equipment & Instruments',
    'Electronic Manufacturing Services',
    # EV & Auto
    'Automobile Manufacturers',
    'Automotive Parts & Equipment',
    'Electrical Components & Equipment',
    # Defense
    'Aerospace & Defense',
    # Clean energy & Industrial
    'Electrical Components & Equipment',
    'Heavy Electrical Equipment',
    'Industrial Machinery & Supplies & Components',
    'Construction Machinery & Heavy Transportation Equipment',
    # Materials — miners of relevant metals
    'Specialty Chemicals',
    'Diversified Metals & Mining',
    'Copper',
    'Aluminum',
    'Steel',
    'Interactive Media & Services',
]

high_exposure_sectors = [
    'Information Technology',  # catches semiconductors broadly
]

# ── TIER 2: MEDIUM EXPECTED EXPOSURE ──────────────────────────────────────────
medium_exposure_subindustries = [
    'Health Care Equipment',        # MRI machines, medical devices
    'Health Care Supplies',
    'Life Sciences Tools & Services',
    'Communications Equipment',     # routers, antennas
    'Technology Hardware & Equipment',
    'Computer Hardware',
    'Data Processing & Outsourced Services',
    'Airport Services',
    'Airlines',                     # aircraft use rare earth components
]

# ── TIER 3: CONTROL GROUP — LOW/NO EXPOSURE ───────────────────────────────────
control_subindustries = [
    'Restaurants',
    'Hotels, Resorts & Cruise Lines',
    'Broadline Retail',
    'Apparel Retail',
    'Food Retail',
    'Hypermarkets & Super Centers',
    'Consumer Finance',
    'Multi-line Insurance',
    'Property & Casualty Insurance',
    'Life & Health Insurance',
    'Diversified Banks',
    'Regional Banks',
    'Investment Banking & Brokerage',
    'Application Software',         # pure software, no hardware
    'IT Consulting & Other Services',
    'Movies & Entertainment',
    'Tobacco',
    'Brewers',
    'Distillers & Vintners',
]

# ── ASSIGN TIERS ──────────────────────────────────────────────────────────────

def assign_tier(row):
    sub = row['GICS Sub-Industry']
    sector = row['GICS Sector']

    if sub in high_exposure_subindustries:
        return 'high'
    if sector == 'Information Technology' and sub in [
        'Semiconductors',
        'Semiconductor Materials & Equipment',
        'Electronic Components',
        'Electronic Equipment & Instruments',
        'Electronic Manufacturing Services',
        'Technology Hardware, Storage & Peripherals',
    ]:
        return 'high'
    if sub in medium_exposure_subindustries:
        return 'medium'
    if sub in control_subindustries:
        return 'control'
    return 'low'

df['tier'] = df.apply(assign_tier, axis=1)

# ── BUILD FINAL SAMPLE ────────────────────────────────────────────────────────

high = df[df['tier'] == 'high']
medium = df[df['tier'] == 'medium']
control = df[df['tier'] == 'control'].head(40)  # cap control at 40

sample = pd.concat([high, medium, control])

print(f"\nSample breakdown:")
print(f"  High exposure  : {len(high)} companies")
print(f"  Medium exposure: {len(medium)} companies")
print(f"  Control group  : {len(control)} companies")
print(f"  TOTAL          : {len(sample)} companies")

print(f"\nHigh exposure companies:")
print(high[['Symbol','Security','GICS Sub-Industry']].to_string())

print(f"\nMedium exposure companies:")
print(medium[['Symbol','Security','GICS Sub-Industry']].to_string())

print(f"\nControl group sample:")
print(control[['Symbol','Security','GICS Sub-Industry']].to_string())

# Save the sample
sample.to_csv('event_study_sample.csv', index=False)
print(f"\nSaved to event_study_sample.csv")

# Extract just the tickers as a Python dict for the event study
tickers_dict = dict(zip(sample['Symbol'], sample['Security']))
print(f"\nTICKERS dict preview (first 10):")
for k, v in list(tickers_dict.items())[:10]:
    print(f'    "{k}": "{v}",')