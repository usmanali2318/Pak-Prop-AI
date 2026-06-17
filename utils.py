import re as _re
import pandas as pd
import numpy as np

# ─── Pakistani Area Unit Conversions ───────────────────────────────────────────
MARLA_TO_SQFT   = 272.251
SQYD_TO_SQFT    = 9.0
KANAL_TO_MARLA  = 20.0
ACRE_TO_MARLA   = 160.0

def convert_to_marla(area_value: float, unit: str) -> float:
    """Convert any Pakistani area unit to Marla (standard internal unit)."""
    unit = str(unit).lower().strip()
    if 'kanal' in unit:
        return area_value * KANAL_TO_MARLA
    elif 'marla' in unit:
        return area_value
    elif 'sq. yd' in unit or 'sq yd' in unit or 'sqyd' in unit or 'square yard' in unit:
        return (area_value * SQYD_TO_SQFT) / MARLA_TO_SQFT
    elif 'sq. ft' in unit or 'sq ft' in unit or 'sqft' in unit or 'square feet' in unit:
        return area_value / MARLA_TO_SQFT
    elif 'sq. m' in unit or 'sqm' in unit or 'square meter' in unit:
        return (area_value * 10.7639) / MARLA_TO_SQFT
    elif 'acre' in unit:
        return area_value * ACRE_TO_MARLA
    else:
        return area_value  # assume marla if unknown

def parse_area_string(area_str: str) -> tuple:
    """Parse area strings like '8 Marla', '2 Kanal', '500 Sq. Yd.' into (value, unit)."""
    if pd.isna(area_str):
        return (np.nan, 'Marla')
    area_str = str(area_str).strip()
    import re
    nums = re.findall(r'[\d,]+\.?\d*', area_str.replace(',', ''))
    value = float(nums[0]) if nums else np.nan
    area_lower = area_str.lower()
    if 'kanal' in area_lower:
        unit = 'Kanal'
    elif 'sq. yd' in area_lower or 'sq yd' in area_lower:
        unit = 'Sq. Yd.'
    elif 'sq. ft' in area_lower or 'sqft' in area_lower:
        unit = 'Sq. Ft.'
    elif 'sq. m' in area_lower or 'sqm' in area_lower:
        unit = 'Sq. M.'
    elif 'acre' in area_lower:
        unit = 'Acre'
    else:
        unit = 'Marla'
    return (value, unit)

def pkr_to_display(amount_pkr: float) -> str:
    """Format PKR amount into human-readable Lakh/Crore format."""
    if pd.isna(amount_pkr) or amount_pkr <= 0:
        return "N/A"
    if amount_pkr >= 1e7:
        crore = amount_pkr / 1e7
        if crore >= 100:
            return f"PKR {crore:.0f} Crore"
        return f"PKR {crore:.2f} Crore"
    elif amount_pkr >= 1e5:
        lakh = amount_pkr / 1e5
        return f"PKR {lakh:.1f} Lakh"
    else:
        return f"PKR {amount_pkr:,.0f}"

def get_area_category(area_sqyd: float) -> str:
    """Categorize property by area in square yards."""
    if pd.isna(area_sqyd):
        return "Unknown"
    elif area_sqyd < 200:
        return "< 200 Sq. Yd."
    elif area_sqyd < 500:
        return "200–500 Sq. Yd."
    elif area_sqyd < 1000:
        return "500–1000 Sq. Yd."
    elif area_sqyd < 2000:
        return "1000–2000 Sq. Yd."
    else:
        return "> 2000 Sq. Yd."

AREA_CAT_ORDER = [
    "< 200 Sq. Yd.",
    "200–500 Sq. Yd.",
    "500–1000 Sq. Yd.",
    "1000–2000 Sq. Yd.",
    "> 2000 Sq. Yd."
]

# ─── City & Property Config ─────────────────────────────────────────────────
SUPPORTED_CITIES = ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Peshawar']

PROPERTY_TYPES = ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'Farm House', 'Room', 'Penthouse']

AREA_UNITS = ['Marla', 'Kanal', 'Sq. Yd.', 'Sq. Ft.']

# Green theme Plotly colors — primary updated to #059669 for better contrast on white
GREEN_PALETTE = ['#065f46', '#059669', '#10b981', '#34d399', '#6ee7b7',
                 '#a7f3d0', '#d1fae5', '#1B5E20', '#2E7D32', '#388E3C']
GREEN_SINGLE  = '#059669'
GREEN_LIGHT   = '#10b981'
GREEN_ACCENT  = '#34d399'

def normalize_location(loc: str) -> str:
    """
    Normalize a raw location string to a canonical form.

    Handles all documented duplicate patterns across the Zameen datasets:
    - Strips trailing city/province suffixes: ", Karachi, Sindh" -> removed
    - Normalizes Urdu connector: "Gulshan E Iqbal" -> "gulshan-e-iqbal"
    - Normalizes phase formatting: "DHA Phase-5", "DHA Phase I" -> "dha phase 5"
    - Normalizes sector codes: "F7", "F 7", "F-7" -> "f-7"
    - Normalizes abbreviations: "P.E.C.H.S" -> "pechs"
    - Normalizes spelling variants: "Teachers Colony" -> "teacher colony"
    - Normalizes common spelling errors: "Baharia" -> "bahria", "defense" -> "defence"

    Returns the normalized string, all lowercase, stripped.
    Returns empty string if input is not a valid string.
    """
    if not isinstance(loc, str) or not loc.strip():
        return ""

    # 1. Strip trailing city / city+province suffix
    loc = _re.sub(
        r',\s*(Karachi|Lahore|Islamabad|Rawalpindi|Faisalabad|Peshawar)(\s*,.*)?$',
        '', loc, flags=_re.IGNORECASE
    ).strip()

    # 2. Lowercase
    loc = loc.lower().strip()

    # 3. Normalize Urdu connector "-e-": "Gulshan E Iqbal" -> "gulshan-e-iqbal"
    loc = _re.sub(r'(?<=[a-z]) e (?=[a-z])', '-e-', loc)

    # 4. Strip punctuation except hyphens
    loc = _re.sub(r"['\",`\.\(\)]", '', loc)

    # 5. Normalize spaces around hyphens: "phase - 5" -> "phase-5"
    loc = _re.sub(r'\s+-\s+', '-', loc)

    # 6. Collapse multiple spaces
    loc = _re.sub(r'\s+', ' ', loc).strip()

    # 7. Known abbreviation expansions
    abbreviations = {
        'p.e.c.h.s': 'pechs',
        'fb area':    'federal b area',
        'f.b area':   'federal b area',
        'f.b. area':  'federal b area',
    }
    for abbr, full in abbreviations.items():
        loc = loc.replace(abbr, full)

    # 8. Sector code normalization: F7 -> f-7, F 7 -> f-7
    loc = _re.sub(r'\b([a-z])(\d+)\b', r'\1-\2', loc)
    loc = _re.sub(r'\b([a-z]) (\d)\b',  r'\1-\2', loc)

    # 9. Phase formatting: "phase-5" -> "phase 5"
    loc = _re.sub(r'\bphase-(\d)', r'phase \1', loc)

    # 10. Roman numeral phases -> Arabic (longer patterns first to avoid partial matches)
    roman_to_arabic = [
        (r'\bphase\s+viii\b', 'phase 8'),
        (r'\bphase\s+vii\b',  'phase 7'),
        (r'\bphase\s+vi\b',   'phase 6'),
        (r'\bphase\s+iv\b',   'phase 4'),
        (r'\bphase\s+iii\b',  'phase 3'),
        (r'\bphase\s+ii\b',   'phase 2'),
        (r'\bphase\s+v\b',    'phase 5'),
        (r'\bphase\s+i\b',    'phase 1'),
    ]
    for pattern, replacement in roman_to_arabic:
        loc = _re.sub(pattern, replacement, loc)

    # 11. Plural normalizations
    loc = _re.sub(r"\bteachers'\b|\bteachers\b", 'teacher', loc)
    loc = _re.sub(r'\bsocieties\b', 'society', loc)

    # 12. Common spelling variants
    loc = loc.replace('defense', 'defence')
    loc = loc.replace('baharia', 'bahria')
    loc = loc.replace('johor',   'johar')
    loc = loc.replace('askary',  'askari')

    # 13. Final cleanup
    loc = _re.sub(r'\s+', ' ', loc).strip()
    return loc


PLOTLY_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, system-ui, -apple-system, sans-serif', size=12, color='#374151'),
    colorway=GREEN_PALETTE,
    xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb', tickfont=dict(size=11)),
    margin=dict(l=60, r=30, t=40, b=50),
)
