"""
rules.py — Regex-based classification for SOS requests.
Expanded to catch cases the ML model misclassifies.
Priority: emergency > medical > crowd > translation > navigation
"""

import re

# -------- Emergency patterns (life-threatening) --------
EMERGENCY_PATTERNS = [
    # Arabic
    r"مغمى\s*عليه",
    r"مغمي\s*عليه",
    r"اغماء|إغماء",
    r"فقدت?\s*الوعي",
    r"لا\s*[اأ]ستطيع\s*التنفس",
    r"نزيف\s*شديد|ينزف",
    r"تشنج|صرع",
    r"إسعاف|اسعاف",
    r"فاقد\s*الوعي",
    r"ما\s*يرد",
    r"طاح\s*و",
    r"سقط\s*و",
    r"وقع\s*و.*نزيف",
    r"لا\s*يتنفس",
    r"توقف\s*قلبه",
    r"حالة\s*خطيرة",
    r"يحتضر",
    # English
    r"unconscious|not breathing|heavy bleeding|seizure|chest pain|fainted",
    r"collapsed|not responding|lost consciousness|heart attack|stroke",
    r"fell down.*bleeding|bleeding.*fell",
    r"someone.*died|person.*died",
    r"can'?t breathe|cannot breathe|difficulty breathing",
    r"choking|suffocating",
    r"severe.*pain|extreme.*pain",
    r"ambulance",
    r"CPR|defibrillator",
]

# -------- Medical patterns (non-life-threatening) --------
MEDICAL_PATTERNS = [
    # Arabic
    r"صداع|صداع\s*شديد",
    r"[اأ]شعر\s*بال[دذ]و[اخ]ر|دوخة|دوار",
    r"استفراغ|تقيؤ|غثيان",
    r"إسهال|اسهال",
    r"حرارة|سخونة|حمى",
    r"[اأ]لم\s*في|[اأ]عاني\s*من",
    r"جفاف",
    r"ضربة\s*شمس",
    r"تعب\s*شديد",
    r"بالدوار",
    # English
    r"dizzy|dizziness|feel faint",
    r"headache|migraine",
    r"vomiting|nausea|throwing up",
    r"diarrhea|stomach",
    r"fever|temperature",
    r"dehydrat|heat\s*stroke|sun\s*stroke",
    r"sprain|fracture|broken",
    r"allergic|allergy|rash",
    r"need.*doctor|need.*medic|need.*nurse",
    r"feeling\s*sick|feel\s*sick|unwell",
    r"injured|injury|wound",
]

# -------- Crowd patterns --------
CROWD_PATTERNS = [
    # Arabic
    r"تدافع",
    r"[اإ]زدحام",
    r"زحام|زحمة",
    r"اختناق",
    r"الناس\s*تدفع",
    r"ضغط\s*الناس",
    r"تكدس",
    # English
    r"crowd\s*crush|stampede|crush",
    r"too\s*(much\s*)?crowd|overcrowd|very\s*crowded",
    r"people\s*push|pushing|being\s*pushed",
    r"can'?t\s*move.*crowd|stuck.*crowd",
    r"dangerous.*crowd|crowd.*dangerous",
]

# -------- Translation patterns --------
TRANSLATION_PATTERNS = [
    # Arabic
    r"مترجم|ترجمة|يترجم|ترجم\s*لي",
    r"ما\s*فهمت|لا\s*[اأ]فهم",
    r"لا\s*يفهم\s*لغتي",
    r"يتحدث|يتكلم",
    # English
    r"translat|interpreter",
    r"don'?t\s*understand.*language|doesn'?t\s*understand.*language",
    r"speak.*language|language\s*barrier",
    r"help\s*me\s*(understand|communicate|speak\s*with)",
    r"can'?t\s*communicate",
    r"what.*saying|what.*said",
    r"need.*speak.*with",
]

# -------- Navigation patterns --------
NAVIGATION_PATTERNS = [
    # Arabic
    r"تائه|ضايع[ةه]?|ضعت",
    r"ما\s*لقيت|لا\s*[اأ]جد",
    r"وين\s*(الطريق|المخرج|الفندق|المخيم|البوابة)",
    r"كيف\s*[اأ]وصل|كيف\s*[اأ]رجع|كيف\s*[اأ]ذهب|كيف\s*اذهب",
    r"[اأ]ين\s*(يقع|يوجد|الطريق)",
    # English
    r"lost|can'?t\s*find",
    r"where\s*is|how\s*(do\s*I|to)\s*(get|go|find|reach)",
    r"directions?\s*to",
    r"which\s*way|which\s*gate|which\s*exit",
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
    Priority order: emergency > medical > crowd > translation > navigation
    """
    text = str(text).strip()

    if _match_any(EMERGENCY_PATTERNS, text):
        return {
            "source": "rules_emergency",
            "request_type": "emergency_response",
            "priority": 5,
            "needs_ambulance": True,
        }

    if _match_any(MEDICAL_PATTERNS, text):
        return {
            "source": "rules_medical",
            "request_type": "medical",
            "priority": 4,
            "needs_ambulance": False,
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

    if _match_any(NAVIGATION_PATTERNS, text):
        return {
            "source": "rules_navigation",
            "request_type": "navigation",
            "priority": 3,
            "needs_ambulance": False,
        }

    return None
