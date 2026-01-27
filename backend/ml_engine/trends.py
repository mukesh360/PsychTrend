"""
Trend analysis module for behavioral pattern detection
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


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


def analyze_motivation_trend(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze motivation trend from responses.
    Uses sentiment and achievement-related keywords.
    """
    if not responses:
        return {
            'name': 'Motivation Trend',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Extract motivation indicators
    motivation_keywords = {
        'motivated', 'inspired', 'excited', 'passionate', 'driven',
        'determined', 'goal', 'achieve', 'success', 'accomplished'
    }
    
    scores = []
    for resp in responses:
        sentiment = resp.get('sentiment_score', 0)
        keywords = set(resp.get('keywords', []))
        
        # Base score from sentiment
        score = (sentiment + 1) / 2  # Normalize to 0-1
        
        # Boost for motivation keywords
        motivation_match = len(keywords.intersection(motivation_keywords))
        if motivation_match > 0:
            score = min(1.0, score + 0.1 * motivation_match)
        
        scores.append(score)
    
    # Calculate overall metrics
    avg_score = sum(scores) / len(scores)
    trend = detect_trend_direction(scores)
    
    # Generate description
    if trend['direction'] == 'upward':
        desc = "Shows increasing motivation over time, with growing enthusiasm for goals and achievements."
    elif trend['direction'] == 'downward':
        desc = "Motivation appears to be declining. Consider identifying sources of inspiration and setting smaller, achievable goals."
    else:
        desc = "Maintains steady motivation levels throughout experiences."
    
    return {
        'name': 'Motivation Trend',
        'score': round(avg_score, 2),
        'trend_direction': trend['direction'],
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'confidence': trend['confidence']
    }


def analyze_consistency(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze behavioral consistency from responses.
    """
    if not responses:
        return {
            'name': 'Consistency Score',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
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
        
        # Start with moderate score
        score = 0.5
        
        # Check for consistency indicators
        for kw in consistency_keywords:
            if kw in text:
                score += 0.08
        
        # Check for volatility indicators
        for kw in volatility_keywords:
            if kw in text:
                score -= 0.05
        
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
    
    # Generate description
    if avg_score > 0.7:
        desc = "Demonstrates strong consistency in behavior and routines. Shows reliable patterns."
    elif avg_score > 0.4:
        desc = "Maintains moderate consistency with some variability. Balanced between routine and flexibility."
    else:
        desc = "Shows high variability in patterns. May benefit from establishing more consistent routines."
    
    return {
        'name': 'Consistency Score',
        'score': round(avg_score, 2),
        'trend_direction': 'stable',
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'volatility': round(volatility_penalty, 2) if len(sentiments) > 1 else 0
    }


def analyze_growth_orientation(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze growth mindset and orientation.
    """
    if not responses:
        return {
            'name': 'Growth Orientation',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    growth_keywords = {
        'learned', 'growth', 'improved', 'developed', 'grew', 'progress',
        'challenge', 'opportunity', 'new', 'skill', 'knowledge', 'better',
        'evolved', 'adapted', 'overcame', 'self-improvement'
    }
    
    fixed_keywords = {
        'stuck', 'can\'t', 'impossible', 'always been', 'never could',
        'born', 'natural', 'talent', 'gifted'
    }
    
    scores = []
    indicators = []
    
    for resp in responses:
        text = resp.get('raw_response', '').lower()
        keywords = resp.get('keywords', [])
        
        score = 0.5
        
        # Check for growth indicators
        for kw in growth_keywords:
            if kw in text:
                score += 0.07
                if kw not in indicators:
                    indicators.append(kw)
        
        # Check for fixed mindset indicators
        for kw in fixed_keywords:
            if kw in text:
                score -= 0.1
        
        scores.append(max(0, min(1, score)))
    
    avg_score = sum(scores) / len(scores)
    trend = detect_trend_direction(scores)
    
    # Generate description
    if avg_score > 0.7:
        desc = "Shows strong growth orientation with focus on learning and development."
    elif avg_score > 0.4:
        desc = "Demonstrates balanced approach to growth with openness to learning."
    else:
        desc = "May benefit from adopting more growth-oriented perspectives."
    
    return {
        'name': 'Growth Orientation',
        'score': round(avg_score, 2),
        'trend_direction': trend['direction'],
        'description': desc,
        'data_points': [round(s, 2) for s in scores],
        'indicators': indicators[:5]  # Top 5 indicators
    }


def analyze_stress_response(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze stress response patterns.
    """
    if not responses:
        return {
            'name': 'Stress Response',
            'score': 0.5,
            'trend_direction': 'stable',
            'description': 'Insufficient data for analysis',
            'data_points': []
        }
    
    # Active coping indicators
    active_coping = {
        'handled', 'managed', 'solved', 'addressed', 'faced', 'overcame',
        'dealt', 'took action', 'worked through', 'found solution'
    }
    
    # Avoidance indicators
    avoidance = {
        'avoided', 'ignored', 'gave up', 'quit', 'walked away',
        'couldn\'t handle', 'too much'
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
        
        coping_scores.append(max(0, min(1, score)))
    
    # Determine dominant pattern
    if active_count > support_count and active_count > avoidance_count:
        pattern = 'active-coping'
        desc = "Tends to address challenges directly with problem-solving approach."
    elif support_count > active_count and support_count > avoidance_count:
        pattern = 'support-seeking'
        desc = "Values collaboration and seeks help when facing challenges."
    elif avoidance_count > 0:
        pattern = 'avoidance-prone'
        desc = "May benefit from developing more direct coping strategies."
    else:
        pattern = 'balanced'
        desc = "Shows balanced approach to handling stressful situations."
    
    avg_score = sum(coping_scores) / len(coping_scores) if coping_scores else 0.5
    
    return {
        'name': 'Stress Response',
        'score': round(avg_score, 2),
        'pattern': pattern,
        'trend_direction': 'stable',
        'description': desc,
        'data_points': [round(s, 2) for s in coping_scores]
    }


def get_all_trends(responses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Get all trend analyses"""
    return {
        'motivation': analyze_motivation_trend(responses),
        'consistency': analyze_consistency(responses),
        'growth': analyze_growth_orientation(responses),
        'stress_response': analyze_stress_response(responses)
    }
