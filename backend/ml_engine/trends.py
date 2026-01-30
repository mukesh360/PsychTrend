"""
Trend analysis module for behavioral pattern detection
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ml_engine.sentiment_context import (
    analyze_sentiment_context,
    get_score_caps,
    STRESS_KEYWORDS,
    LOW_MOTIVATION_KEYWORDS,
    UNCERTAINTY_KEYWORDS,
    FEAR_DISCOURAGEMENT_KEYWORDS
)


def calculate_moving_average(values: List[float], window: int = 3) -> List[float]:
    """Calculate moving average with specified window"""
    if len(values) < window:
        return values
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i+1]
        result.append(sum(window_values) / len(window_values))
    
    return result


def detect_trend_direction(values: List[float]) -> Dict[str, Any]:
    """
    Detect overall trend direction using linear regression.
    """
    if len(values) < 2:
        return {
            'direction': 'insufficient_data',
            'slope': 0.0,
            'confidence': 0.0
        }
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Linear regression
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    
    # Calculate R-squared for confidence
    y_pred = np.polyval(coefficients, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine direction
    if slope > 0.02:
        direction = 'upward'
    elif slope < -0.02:
        direction = 'downward'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'slope': round(float(slope), 4),
        'confidence': round(float(max(0, r_squared)), 3)
    }


def detect_change_points(values: List[float], threshold: float = 0.3) -> List[int]:
    """
    Detect significant change points in the data.
    Returns indices where significant changes occur.
    """
    if len(values) < 3:
        return []
    
    change_points = []
    
    for i in range(1, len(values) - 1):
        # Calculate local change
        before = values[i-1]
        current = values[i]
        after = values[i+1]
        
        # Check for significant change
        change_before = abs(current - before)
        change_after = abs(after - current)
        
        if change_before > threshold or change_after > threshold:
            change_points.append(i)
    
    return change_points


def analyze_motivation_trend(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze motivation trend from responses.
    Uses sentiment and achievement-related keywords.
    Applies score caps when negative sentiment dominates.
    """
    if not responses:
        return {
            'name': 'Motivation Trend',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Get score caps based on sentiment
    score_caps = get_score_caps(sentiment_context)
    max_score = score_caps.get('motivation', 1.0)
    
    # Extract motivation indicators
    motivation_keywords = {
        'motivated', 'inspired', 'excited', 'passionate', 'driven',
        'determined', 'goal', 'achieve', 'success', 'accomplished'
    }
    
    # Combine all text for negative indicator detection
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    
    # Detect negative motivation indicators
    stress_indicators = sum(1 for kw in STRESS_KEYWORDS if kw in all_text)
    low_motivation_indicators = sum(1 for kw in LOW_MOTIVATION_KEYWORDS if kw in all_text)
    
    scores = []
    for resp in responses:
        sentiment = resp.get('sentiment_score', 0)
        keywords = set(resp.get('keywords', []))
        input_quality = resp.get('input_quality', 1.0)
        resp_text = resp.get('raw_response', '').lower()
        
        # Base score from sentiment
        score = (sentiment + 1) / 2  # Normalize to 0-1
        
        # Boost for motivation keywords
        motivation_match = len(keywords.intersection(motivation_keywords))
        if motivation_match > 0:
            score = min(1.0, score + 0.1 * motivation_match)
        
        # STRICT: Penalty for stress/pressure language in this response
        if any(kw in resp_text for kw in STRESS_KEYWORDS):
            score = score * 0.6
        
        # STRICT: Penalty for low motivation language
        if any(kw in resp_text for kw in LOW_MOTIVATION_KEYWORDS):
            score = score * 0.5
        
        # Apply quality penalty for low-quality inputs
        if input_quality < 0.5:
            # Reduce score significantly for poor responses
            score = score * input_quality * 0.6
        
        scores.append(max(0, min(1, score)))
    
    # Calculate overall metrics
    avg_score = sum(scores) / len(scores)
    
    # STRICT: Apply score cap based on sentiment context
    avg_score = min(avg_score, max_score)
    
    # Additional reduction for multiple negative indicators
    if stress_indicators >= 2 or low_motivation_indicators >= 1:
        avg_score = min(avg_score, 0.45)
    if stress_indicators >= 3 or low_motivation_indicators >= 2:
        avg_score = min(avg_score, 0.35)
    
    trend = detect_trend_direction(scores)
    
    # Generate description - tone appropriate to score
    if avg_score >= 0.7:
        desc = "Shows increasing motivation over time, with growing enthusiasm for goals and achievements."
    elif avg_score >= 0.45:
        desc = "Shows moderate motivation with some variability. May be experiencing periods of reduced drive."
    elif avg_score >= 0.3:
        desc = "Currently experiencing challenges with motivation. Shows signs of stress or pressure affecting drive."
    else:
        desc = "May be facing significant motivation challenges. Shows indicators of stress, pressure, or discouragement."
    
    return {
        'name': 'Motivation Trend',
        'score': round(avg_score, 2),
        'trend_direction': trend['direction'],
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'confidence': trend['confidence'],
        'score_capped': avg_score < (sum(scores) / len(scores)) if scores else False
    }


def analyze_consistency(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze behavioral consistency from responses.
    Applies score caps when negative sentiment dominates.
    """
    if not responses:
        return {
            'name': 'Consistency Score',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Get score caps based on sentiment
    score_caps = get_score_caps(sentiment_context)
    max_score = score_caps.get('consistency', 1.0)
    
    # Combine all text for negative indicator detection
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    
    # Detect overwhelming/exhausting routine descriptions
    exhaustion_keywords = {'tiring', 'exhausting', 'overwhelming', 'draining', 'too much', 'burnt out'}
    exhaustion_count = sum(1 for kw in exhaustion_keywords if kw in all_text)
    
    # Look for consistency indicators
    consistency_keywords = {
        'routine', 'habit', 'regular', 'daily', 'always', 'consistent',
        'discipline', 'practice', 'maintain', 'steady'
    }
    
    volatility_keywords = {
        'change', 'different', 'varies', 'sometimes', 'unpredictable',
        'spontaneous', 'flexible'
    }
    
    scores = []
    for resp in responses:
        keywords = set(k.lower() for k in resp.get('keywords', []))
        text = resp.get('raw_response', '').lower()
        input_quality = resp.get('input_quality', 1.0)
        
        # STRICT: Start with lower base for low-quality inputs
        base_score = 0.5 if input_quality >= 0.5 else 0.15
        score = base_score
        
        # Check for consistency indicators
        for kw in consistency_keywords:
            if kw in text:
                score += 0.08
        
        # Check for volatility indicators
        for kw in volatility_keywords:
            if kw in text:
                score -= 0.05
        
        # STRICT: Penalty for exhaustion/overwhelming language
        if any(kw in text for kw in exhaustion_keywords):
            score = score * 0.4
        
        # Apply quality penalty
        if input_quality < 0.5:
            score = score * input_quality * 0.5
        
        scores.append(max(0, min(1, score)))
    
    # Calculate volatility from sentiment changes
    sentiments = [r.get('sentiment_score', 0) for r in responses]
    if len(sentiments) > 1:
        changes = [abs(sentiments[i] - sentiments[i-1]) for i in range(1, len(sentiments))]
        avg_change = sum(changes) / len(changes)
        volatility_penalty = min(0.3, avg_change)
    else:
        volatility_penalty = 0
    
    avg_score = max(0, sum(scores) / len(scores) - volatility_penalty)
    
    # STRICT: Apply score cap based on sentiment context
    avg_score = min(avg_score, max_score)
    
    # Additional reduction for exhaustion indicators
    if exhaustion_count >= 1:
        avg_score = min(avg_score, 0.30)
    if exhaustion_count >= 2:
        avg_score = min(avg_score, 0.20)
    
    # Generate description - tone appropriate to score
    if avg_score > 0.7:
        desc = "Demonstrates strong consistency in behavior and routines. Shows reliable patterns."
    elif avg_score > 0.4:
        desc = "Maintains moderate consistency with some variability. Balanced between routine and flexibility."
    elif avg_score > 0.25:
        desc = "Currently experiencing difficulty maintaining consistent patterns. Routines may feel overwhelming or tiring."
    else:
        desc = "Shows signs of struggling with consistency. May be facing challenges in maintaining routines due to stress or exhaustion."
    
    return {
        'name': 'Consistency Score',
        'score': round(avg_score, 2),
        'trend_direction': 'stable',
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'volatility': round(volatility_penalty, 2) if len(sentiments) > 1 else 0,
        'score_capped': avg_score < (sum(scores) / len(scores) - volatility_penalty) if scores else False
    }


def analyze_growth_orientation(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze growth mindset and orientation.
    Applies score caps when negative sentiment dominates.
    """
    if not responses:
        return {
            'name': 'Growth Orientation',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Get score caps based on sentiment
    score_caps = get_score_caps(sentiment_context)
    max_score = score_caps.get('growth', 1.0)
    
    # Combine all text for negative indicator detection
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    
    # Detect uncertainty about growth indicators
    uncertainty_indicators = sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in all_text)
    fear_indicators = sum(1 for kw in FEAR_DISCOURAGEMENT_KEYWORDS if kw in all_text)
    
    # Keywords suggesting growth feels forced or unclear
    forced_growth_keywords = {'forced', 'have to', 'must', 'pressure', 'expected', 'should'}
    forced_count = sum(1 for kw in forced_growth_keywords if kw in all_text)
    
    growth_keywords = {
        'learned', 'growth', 'improved', 'developed', 'grew', 'progress',
        'challenge', 'opportunity', 'new', 'skill', 'knowledge', 'better',
        'evolved', 'adapted', 'overcame', 'self-improvement'
    }
    
    fixed_keywords = {
        'stuck', "can't", 'impossible', 'always been', 'never could',
        'born', 'natural', 'talent', 'gifted', 'hopeless', 'pointless'
    }
    
    scores = []
    indicators = []
    
    for resp in responses:
        text = resp.get('raw_response', '').lower()
        keywords = resp.get('keywords', [])
        input_quality = resp.get('input_quality', 1.0)
        
        # STRICT: Start with lower base score for low-quality inputs
        score = 0.5 if input_quality >= 0.5 else 0.10
        
        # Check for growth indicators
        for kw in growth_keywords:
            if kw in text:
                score += 0.07
                if kw not in indicators:
                    indicators.append(kw)
        
        # Check for fixed mindset indicators
        for kw in fixed_keywords:
            if kw in text:
                score -= 0.12  # STRICT: Increased penalty
        
        # STRICT: Penalty for uncertainty language
        if any(kw in text for kw in UNCERTAINTY_KEYWORDS):
            score = score * 0.7
        
        # STRICT: Penalty for fear/discouragement language
        if any(kw in text for kw in FEAR_DISCOURAGEMENT_KEYWORDS):
            score = score * 0.5
        
        # Apply quality penalty
        if input_quality < 0.5:
            score = score * input_quality * 0.5
        
        scores.append(max(0, min(1, score)))
    
    avg_score = sum(scores) / len(scores)
    
    # STRICT: Apply score cap based on sentiment context
    avg_score = min(avg_score, max_score)
    
    # Additional reduction for uncertainty/forced growth
    if uncertainty_indicators >= 2 or forced_count >= 2:
        avg_score = min(avg_score, 0.45)
    if fear_indicators >= 1:
        avg_score = min(avg_score, 0.40)
    if uncertainty_indicators >= 3 or fear_indicators >= 2:
        avg_score = min(avg_score, 0.30)
    
    trend = detect_trend_direction(scores)
    
    # Generate description - tone appropriate to score
    if avg_score > 0.7:
        desc = "Shows strong growth orientation with focus on learning and development."
    elif avg_score > 0.45:
        desc = "Demonstrates moderate openness to growth. May benefit from clearer direction."
    elif avg_score > 0.3:
        desc = "Currently experiencing uncertainty about growth. Growth direction may feel unclear or forced."
    else:
        desc = "May be facing challenges with growth orientation. Shows signs of uncertainty, discouragement, or feeling stuck."
    
    return {
        'name': 'Growth Orientation',
        'score': round(avg_score, 2),
        'trend_direction': trend['direction'],
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'indicators': indicators[:5],  # Top 5 indicators
        'score_capped': avg_score < (sum(scores) / len(scores)) if scores else False
    }


def analyze_stress_response(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze stress response patterns.
    Applies score caps when negative sentiment dominates.
    """
    if not responses:
        return {
            'name': 'Stress Response',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Get score caps based on sentiment
    score_caps = get_score_caps(sentiment_context)
    max_score = score_caps.get('stress_response', 1.0)
    
    # Combine all text for negative indicator detection
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    
    # Detect fear/discouragement indicators
    fear_indicators = sum(1 for kw in FEAR_DISCOURAGEMENT_KEYWORDS if kw in all_text)
    stress_indicators = sum(1 for kw in STRESS_KEYWORDS if kw in all_text)
    
    # Active coping indicators
    active_coping = {
        'handled', 'managed', 'solved', 'addressed', 'faced', 'overcame',
        'dealt', 'took action', 'worked through', 'found solution'
    }
    
    # Avoidance indicators - expanded
    avoidance = {
        'avoided', 'ignored', 'gave up', 'quit', 'walked away',
        "couldn't handle", 'too much', 'ran away', 'escaped',
        'shut down', 'frozen', 'paralyzed', 'overwhelmed'
    }
    
    # Support-seeking indicators
    support_seeking = {
        'help', 'support', 'talked', 'asked', 'reached out', 'team',
        'family', 'friends', 'mentor'
    }
    
    coping_scores = []
    pattern = 'balanced'
    
    active_count = 0
    support_count = 0
    avoidance_count = 0
    
    for resp in responses:
        text = resp.get('raw_response', '').lower()
        sentiment = resp.get('sentiment_score', 0)
        input_quality = resp.get('input_quality', 1.0)
        
        # Analyze coping style
        for kw in active_coping:
            if kw in text:
                active_count += 1
        
        for kw in support_seeking:
            if kw in text:
                support_count += 1
        
        for kw in avoidance:
            if kw in text:
                avoidance_count += 1
        
        # Calculate resilience score
        if any(kw in text for kw in active_coping):
            score = 0.7 + (sentiment + 1) / 10
        elif any(kw in text for kw in support_seeking):
            score = 0.6 + (sentiment + 1) / 10
        else:
            score = 0.5 + (sentiment + 1) / 10
        
        # STRICT: Penalty for avoidance indicators
        if any(kw in text for kw in avoidance):
            score = score * 0.5
        
        # STRICT: Penalty for fear/discouragement in this response
        if any(kw in text for kw in FEAR_DISCOURAGEMENT_KEYWORDS):
            score = score * 0.6
        
        # STRICT: Quality penalty
        if input_quality < 0.5:
            score = score * input_quality * 0.6
        
        coping_scores.append(max(0, min(1, score)))
    
    avg_score = sum(coping_scores) / len(coping_scores) if coping_scores else 0.5
    
    # STRICT: Apply score cap based on sentiment context
    avg_score = min(avg_score, max_score)
    
    # Additional reduction for fear/discouragement indicators
    if fear_indicators >= 1 or stress_indicators >= 2:
        avg_score = min(avg_score, 0.45)
    if fear_indicators >= 2 or avoidance_count >= 2:
        avg_score = min(avg_score, 0.35)
    if fear_indicators >= 3 or (stress_indicators >= 2 and fear_indicators >= 1):
        avg_score = min(avg_score, 0.25)
    
    # Determine dominant pattern
    if avoidance_count > active_count and avoidance_count > support_count:
        pattern = 'avoidance-prone'
        desc = "Currently showing signs of difficulty handling stress. May benefit from developing coping strategies."
    elif active_count > support_count and active_count > avoidance_count:
        pattern = 'active-coping'
        desc = "Tends to address challenges directly with problem-solving approach."
    elif support_count > active_count and support_count > avoidance_count:
        pattern = 'support-seeking'
        desc = "Values collaboration and seeks help when facing challenges."
    else:
        pattern = 'balanced'
        if avg_score < 0.4:
            desc = "May be facing challenges with stress management. Shows signs of discouragement or fear."
        else:
            desc = "Shows balanced approach to handling stressful situations."
    
    return {
        'name': 'Stress Response',
        'score': round(avg_score, 2),
        'pattern': pattern,
        'trend_direction': 'stable',
        'description': desc,
        'data_points': [round(s, 2) for s in coping_scores],
        'score_capped': avg_score < (sum(coping_scores) / len(coping_scores)) if coping_scores else False
    }


def get_all_trends(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all trend analyses with sentiment context for stricter scoring.
    
    Args:
        responses: List of structured response dicts
        sentiment_context: Optional pre-computed sentiment context
        
    Returns:
        Dict with all trend analyses including sentiment_context
    """
    # Compute sentiment context once and pass to all functions
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    return {
        'motivation': analyze_motivation_trend(responses, sentiment_context),
        'consistency': analyze_consistency(responses, sentiment_context),
        'growth': analyze_growth_orientation(responses, sentiment_context),
        'stress_response': analyze_stress_response(responses, sentiment_context),
        'sentiment_context': sentiment_context  # Include for report generation
    }
