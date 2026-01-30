"""
Sentiment Context Analysis for Stricter Behavioral Scoring

This module detects when negative sentiment dominates user responses
and provides score caps and archetype restrictions accordingly.
"""
from typing import Dict, List, Any, Tuple, Set


# =============================================================================
# NEGATIVE INDICATOR KEYWORDS
# =============================================================================

# Uncertainty and confusion indicators
UNCERTAINTY_KEYWORDS = {
    "don't know", "not sure", "uncertain", "confused", "unclear", "maybe",
    "perhaps", "idk", "dunno", "no idea", "unsure", "hard to say",
    "can't tell", "I guess", "possibly", "doubtful", "hesitant"
}

# Stress and pressure indicators
STRESS_KEYWORDS = {
    "stressed", "pressure", "overwhelmed", "exhausted", "tired", "burnt out",
    "anxious", "worried", "tense", "overworked", "demanding", "hectic",
    "struggling", "difficult", "hard time", "tough", "draining", "burden"
}

# Fear and discouragement indicators
FEAR_DISCOURAGEMENT_KEYWORDS = {
    "afraid", "scared", "fear", "worried", "discouraged", "hopeless",
    "give up", "gave up", "can't", "cannot", "impossible", "won't work",
    "failed", "failure", "losing", "lost", "stuck", "trapped", "helpless",
    "pointless", "useless", "worthless", "no point"
}

# Low motivation indicators
LOW_MOTIVATION_KEYWORDS = {
    "don't want", "unmotivated", "lazy", "bored", "apathetic", "indifferent",
    "lack of interest", "no motivation", "forced", "have to", "must",
    "obligation", "reluctant", "unwilling", "dread", "hate"
}

# Absence of achievement indicators
NO_ACHIEVEMENT_KEYWORDS = {
    "nothing", "didn't accomplish", "no progress", "haven't done",
    "nothing special", "not much", "same old", "routine", "boring",
    "mundane", "ordinary", "unremarkable", "mediocre", "average"
}

# Positive achievement indicators (for blocking "Achiever" without these)
ACHIEVEMENT_EVIDENCE_KEYWORDS = {
    "achieved", "accomplished", "proud", "success", "won", "earned",
    "completed", "finished", "reached", "goal", "milestone", "breakthrough",
    "recognition", "award", "promoted", "excelled", "best", "first place"
}

# Positive growth indicators (for blocking high growth without these)
GROWTH_EVIDENCE_KEYWORDS = {
    "learned", "grew", "improved", "developed", "progressed", "better",
    "enhanced", "expanded", "evolved", "transformed", "mastered"
}


# =============================================================================
# SENTIMENT CONTEXT ANALYSIS
# =============================================================================

def analyze_sentiment_context(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze overall sentiment context from all responses.
    
    Returns:
        Dict with:
        - is_negative_dominant: bool
        - negative_ratio: float (0.0 to 1.0)
        - uncertainty_count: int
        - stress_count: int
        - fear_count: int
        - low_motivation_count: int
        - no_achievement_count: int
        - has_achievement_evidence: bool
        - has_growth_evidence: bool
        - attention_areas: List[str] - areas impacted by negative sentiment
    """
    if not responses:
        return {
            "is_negative_dominant": False,
            "negative_ratio": 0.0,
            "uncertainty_count": 0,
            "stress_count": 0,
            "fear_count": 0,
            "low_motivation_count": 0,
            "no_achievement_count": 0,
            "has_achievement_evidence": False,
            "has_growth_evidence": False,
            "attention_areas": []
        }
    
    # Combine all response text
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    
    # Count negative indicators
    uncertainty_count = sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in all_text)
    stress_count = sum(1 for kw in STRESS_KEYWORDS if kw in all_text)
    fear_count = sum(1 for kw in FEAR_DISCOURAGEMENT_KEYWORDS if kw in all_text)
    low_motivation_count = sum(1 for kw in LOW_MOTIVATION_KEYWORDS if kw in all_text)
    no_achievement_count = sum(1 for kw in NO_ACHIEVEMENT_KEYWORDS if kw in all_text)
    
    # Check for positive evidence
    has_achievement_evidence = any(kw in all_text for kw in ACHIEVEMENT_EVIDENCE_KEYWORDS)
    has_growth_evidence = any(kw in all_text for kw in GROWTH_EVIDENCE_KEYWORDS)
    
    # Calculate negative sentiment from scores
    sentiment_scores = [r.get('sentiment_score', 0) for r in responses]
    negative_responses = sum(1 for s in sentiment_scores if s < -0.1)
    low_quality_responses = sum(1 for r in responses if r.get('input_quality', 1.0) < 0.5)
    
    total_responses = len(responses)
    negative_ratio = (negative_responses + low_quality_responses) / (total_responses * 2)
    
    # Count total negative indicators
    total_negative_indicators = (
        uncertainty_count + stress_count + fear_count +
        low_motivation_count + no_achievement_count
    )
    
    # Determine if negative sentiment dominates
    is_negative_dominant = (
        negative_ratio > 0.3 or
        total_negative_indicators >= 3 or
        (stress_count >= 2 and fear_count >= 1) or
        (low_motivation_count >= 2)
    )
    
    # Identify attention areas
    attention_areas = []
    if low_motivation_count >= 1 or stress_count >= 2:
        attention_areas.append("motivation stability")
    if uncertainty_count >= 2:
        attention_areas.append("confidence")
    if stress_count >= 1 or fear_count >= 1:
        attention_areas.append("stress handling")
    if fear_count >= 2 or (stress_count >= 1 and fear_count >= 1):
        attention_areas.append("emotional regulation")
    
    return {
        "is_negative_dominant": is_negative_dominant,
        "negative_ratio": round(negative_ratio, 2),
        "uncertainty_count": uncertainty_count,
        "stress_count": stress_count,
        "fear_count": fear_count,
        "low_motivation_count": low_motivation_count,
        "no_achievement_count": no_achievement_count,
        "has_achievement_evidence": has_achievement_evidence,
        "has_growth_evidence": has_growth_evidence,
        "attention_areas": attention_areas
    }


def get_score_caps(sentiment_context: Dict[str, Any]) -> Dict[str, float]:
    """
    Get maximum allowed scores based on sentiment context.
    
    Returns:
        Dict with max scores for each trend:
        - motivation: max 0.45 if negative
        - consistency: max 0.30 if negative
        - growth: max 0.45 if negative
        - stress_response: max 0.45 if negative
    """
    if not sentiment_context.get("is_negative_dominant", False):
        # No caps if not negative dominant
        return {
            "motivation": 1.0,
            "consistency": 1.0,
            "growth": 1.0,
            "stress_response": 1.0
        }
    
    # Apply strict caps for negative sentiment
    caps = {
        "motivation": 0.45,
        "consistency": 0.30,
        "growth": 0.45,
        "stress_response": 0.45
    }
    
    # Further reduce based on specific indicators
    ctx = sentiment_context
    
    # Extra reduction for multiple stress/fear indicators
    if ctx.get("stress_count", 0) >= 3:
        caps["motivation"] = min(caps["motivation"], 0.35)
        caps["stress_response"] = min(caps["stress_response"], 0.35)
    
    if ctx.get("fear_count", 0) >= 2:
        caps["stress_response"] = min(caps["stress_response"], 0.35)
        caps["growth"] = min(caps["growth"], 0.35)
    
    if ctx.get("low_motivation_count", 0) >= 2:
        caps["motivation"] = min(caps["motivation"], 0.30)
    
    if ctx.get("uncertainty_count", 0) >= 3:
        caps["growth"] = min(caps["growth"], 0.35)
        caps["consistency"] = min(caps["consistency"], 0.25)
    
    return caps


def get_blocked_archetypes(sentiment_context: Dict[str, Any]) -> Set[str]:
    """
    Get archetypes that should NOT be assigned given the sentiment context.
    
    Returns:
        Set of archetype names to block
    """
    blocked = set()
    
    # Block "Achiever" unless explicit achievement evidence
    if not sentiment_context.get("has_achievement_evidence", False):
        blocked.add("achiever")
    
    # If negative dominant, block positive archetypes
    if sentiment_context.get("is_negative_dominant", False):
        blocked.add("achiever")
        blocked.add("innovator")  # Requires positive creative evidence
        
        # Block stabilizer if high stress (can't maintain routine under stress)
        if sentiment_context.get("stress_count", 0) >= 2:
            blocked.add("stabilizer")
    
    return blocked


def get_preferred_archetypes(sentiment_context: Dict[str, Any]) -> List[str]:
    """
    Get archetypes that are preferred given negative sentiment context.
    These are non-judgmental, development-focused archetypes.
    
    Returns:
        List of preferred archetype names in priority order
    """
    if not sentiment_context.get("is_negative_dominant", False):
        return []  # No preference override
    
    preferred = []
    
    # Based on specific indicators, suggest appropriate archetypes
    if sentiment_context.get("uncertainty_count", 0) >= 2:
        preferred.append("exploring")
    
    if sentiment_context.get("low_motivation_count", 0) >= 1:
        preferred.append("developing")
    
    if sentiment_context.get("fear_count", 0) >= 1:
        preferred.append("emerging")
    
    # Default for negative sentiment
    if not preferred:
        preferred = ["developing", "exploring"]
    
    return preferred


def get_description_tone_guidance(sentiment_context: Dict[str, Any]) -> str:
    """
    Get guidance for LLM on appropriate tone for descriptions.
    
    Returns:
        String with tone guidance to include in prompts
    """
    if not sentiment_context.get("is_negative_dominant", False):
        return ""
    
    attention_areas = sentiment_context.get("attention_areas", [])
    
    guidance = """
IMPORTANT TONE GUIDANCE (based on detected negative sentiment):
- Use cautious, evidence-based language
- Do NOT apply optimism bias or positive reframing
- Reflect the emotional tone accurately
- Use phrases like: "currently experiencing", "shows signs of", "may be facing challenges with"
- Do NOT interpret stress or pressure as achievement
- Do NOT assume resilience without explicit evidence
"""
    
    if attention_areas:
        guidance += f"\nBehavioral Attention Areas to highlight: {', '.join(attention_areas)}"
    
    return guidance
