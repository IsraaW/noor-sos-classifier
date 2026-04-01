"""
rules.py — Regex-based emergency / crowd / translation detection.
Extracted directly from the Colab notebook so the API stays consistent.
"""

import re

# -------- Emergency patterns --------
EMERGENCY_PATTERNS = [
    r"مغمى\s*عليه",
    r"مغمي\s*عليه",
    r"اغماء|إغماء",
    r"فقدت\s*الوعي|فقد\s*الوعي",
    r"لا\s*أستطيع\s*التنفس|لا\s*استطيع\s*التنفس",
    r"نزيف\s*شديد",
    r"تشنج|صرع",
    r"إسعاف|اسعاف|ambulance",
    r"unconscious|not breathing|heavy bleeding|seizure|chest pain|fainted",
]

# -------- Crowd patterns --------
CROWD_PATTERNS = [
    r"تدافع",
    r"ازدحام\s*خطير",
    r"زحام\s*خطير",
    r"اختناق",
    r"crowd crush|stampede|crush",
]

# -------- Translation patterns --------
TRANSLATION_PATTERNS = [
    r"مترجم|ترجمة|يترجم|ترجم\s*لي|translate|translator|translation",
]

# -------- Default priority by type --------
DEFAULT_PRIORITY = {
    "emergency_response": 5,
    "medical": 4,
    "crowd_management": 4,
    "navigation": 3,
    "translation": 2,
    "general_guidance": 1,
}


def _match_any(patterns, text):
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def rule_check(text: str):
    """
    Returns a dict if a rule fires, otherwise None.
    """
    if _match_any(EMERGENCY_PATTERNS, text):
        return {
            "source": "rules_emergency",
            "request_type": "emergency_response",
            "priority": 5,
            "needs_ambulance": True,
        }

    if _match_any(CROWD_PATTERNS, text):
        return {
            "source": "rules_crowd",
            "request_type": "crowd_management",
            "priority": 4,
            "needs_ambulance": False,
        }

    if _match_any(TRANSLATION_PATTERNS, text):
        return {
            "source": "rules_translation",
            "request_type": "translation",
            "priority": 2,
            "needs_ambulance": False,
        }

    return None
