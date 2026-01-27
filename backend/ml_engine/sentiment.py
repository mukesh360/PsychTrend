"""
Sentiment analysis module using rule-based and ML approaches
"""
import re
from typing import Dict, List, Tuple, Any
import numpy as np


# Extended sentiment lexicons
POSITIVE_LEXICON = {
    # High intensity (score: 0.8-1.0)
    'amazing': 0.9, 'wonderful': 0.9, 'excellent': 0.9, 'fantastic': 0.9,
    'incredible': 0.9, 'outstanding': 0.9, 'brilliant': 0.9, 'exceptional': 0.9,
    'thrilled': 0.95, 'ecstatic': 0.95, 'overjoyed': 0.95,
    
    # Medium intensity (score: 0.5-0.8)
    'happy': 0.7, 'glad': 0.7, 'pleased': 0.7, 'satisfied': 0.7,
    'proud': 0.75, 'confident': 0.7, 'motivated': 0.7, 'inspired': 0.75,
    'successful': 0.8, 'accomplished': 0.8, 'achieved': 0.8,
    'love': 0.8, 'passionate': 0.8, 'enjoy': 0.7, 'excited': 0.75,
    'grateful': 0.75, 'thankful': 0.75, 'blessed': 0.7,
    'resilient': 0.7, 'determined': 0.7, 'focused': 0.65,
    'creative': 0.65, 'innovative': 0.65, 'productive': 0.65,
    
    # Low intensity (score: 0.3-0.5)
    'good': 0.5, 'nice': 0.5, 'fine': 0.4, 'okay': 0.35, 'ok': 0.35,
    'better': 0.5, 'improved': 0.55, 'learned': 0.5, 'grew': 0.55,
    'interesting': 0.45, 'helpful': 0.5, 'useful': 0.45,
    'comfortable': 0.5, 'calm': 0.5, 'relaxed': 0.55
}

NEGATIVE_LEXICON = {
    # High intensity (score: -0.8 to -1.0)
    'terrible': -0.9, 'horrible': -0.9, 'awful': -0.9, 'devastating': -0.95,
    'miserable': -0.9, 'hopeless': -0.9, 'desperate': -0.85,
    'traumatic': -0.95, 'unbearable': -0.9,
    
    # Medium intensity (score: -0.5 to -0.8)
    'sad': -0.7, 'unhappy': -0.7, 'depressed': -0.8, 'disappointed': -0.7,
    'frustrated': -0.7, 'angry': -0.75, 'upset': -0.65,
    'stressed': -0.7, 'anxious': -0.7, 'worried': -0.65, 'nervous': -0.6,
    'failed': -0.75, 'failure': -0.75, 'rejected': -0.7,
    'overwhelmed': -0.7, 'exhausted': -0.65, 'burned': -0.7,
    'struggled': -0.6, 'difficult': -0.55, 'challenging': -0.5,
    'lonely': -0.7, 'isolated': -0.7, 'hurt': -0.7,
    
    # Low intensity (score: -0.3 to -0.5)
    'bad': -0.5, 'hard': -0.45, 'tough': -0.45, 'problem': -0.4,
    'issue': -0.35, 'concern': -0.35, 'doubt': -0.45,
    'confused': -0.45, 'uncertain': -0.45, 'stuck': -0.5,
    'tired': -0.4, 'bored': -0.35
}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'hardly', 'barely', 'without'}
INTENSIFIERS = {'very': 1.3, 'really': 1.3, 'extremely': 1.5, 'incredibly': 1.5, 
                'absolutely': 1.5, 'totally': 1.3, 'completely': 1.4}
DIMINISHERS = {'somewhat': 0.6, 'slightly': 0.5, 'a bit': 0.6, 'kind of': 0.6, 
               'sort of': 0.6, 'barely': 0.4, 'hardly': 0.4}


def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    return re.findall(r"[\w']+|[.,!?;]", text.lower())


def analyze_sentiment_detailed(text: str) -> Dict[str, Any]:
    """
    Perform detailed sentiment analysis.
    
    Returns:
        - score: float (-1 to 1)
        - magnitude: float (0 to 1, strength of sentiment)
        - category: str (positive, negative, neutral)
        - contributing_words: list of (word, score) tuples
    """
    if not text:
        return {
            'score': 0.0,
            'magnitude': 0.0,
            'category': 'neutral',
            'contributing_words': []
        }
    
    tokens = tokenize(text)
    word_scores = []
    contributing_words = []
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        base_score = 0.0
        
        # Check lexicons
        if token in POSITIVE_LEXICON:
            base_score = POSITIVE_LEXICON[token]
        elif token in NEGATIVE_LEXICON:
            base_score = NEGATIVE_LEXICON[token]
        else:
            i += 1
            continue
        
        # Check for modifiers (look back 2 words)
        modifier = 1.0
        is_negated = False
        
        for j in range(max(0, i-2), i):
            prev_token = tokens[j]
            if prev_token in NEGATION_WORDS or "'t" in prev_token:
                is_negated = True
            if prev_token in INTENSIFIERS:
                modifier = INTENSIFIERS[prev_token]
            elif prev_token in DIMINISHERS:
                modifier = DIMINISHERS[prev_token]
        
        # Apply modifiers
        final_score = base_score * modifier
        if is_negated:
            final_score = -final_score * 0.7  # Negation flips and slightly reduces
        
        word_scores.append(final_score)
        contributing_words.append((token, round(final_score, 3)))
        
        i += 1
    
    # Calculate overall score
    if word_scores:
        raw_score = sum(word_scores) / len(word_scores)
        magnitude = sum(abs(s) for s in word_scores) / len(word_scores)
    else:
        raw_score = 0.0
        magnitude = 0.0
    
    # Normalize
    score = max(-1.0, min(1.0, raw_score))
    
    # Categorize
    if score > 0.15:
        category = 'positive'
    elif score < -0.15:
        category = 'negative'
    else:
        category = 'neutral'
    
    return {
        'score': round(score, 3),
        'magnitude': round(magnitude, 3),
        'category': category,
        'contributing_words': contributing_words
    }


def analyze_sentiment_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Analyze sentiment for multiple texts"""
    return [analyze_sentiment_detailed(text) for text in texts]


def calculate_sentiment_trend(sentiment_scores: List[float]) -> Dict[str, Any]:
    """
    Calculate trend from a series of sentiment scores.
    
    Returns:
        - direction: str (upward, downward, stable)
        - slope: float
        - volatility: float
        - average: float
    """
    if not sentiment_scores:
        return {
            'direction': 'stable',
            'slope': 0.0,
            'volatility': 0.0,
            'average': 0.0
        }
    
    scores = np.array(sentiment_scores)
    n = len(scores)
    
    # Calculate average
    average = float(np.mean(scores))
    
    # Calculate volatility (standard deviation)
    volatility = float(np.std(scores)) if n > 1 else 0.0
    
    # Calculate slope using linear regression
    if n > 1:
        x = np.arange(n)
        slope = float(np.polyfit(x, scores, 1)[0])
    else:
        slope = 0.0
    
    # Determine direction
    if slope > 0.05:
        direction = 'upward'
    elif slope < -0.05:
        direction = 'downward'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'slope': round(slope, 4),
        'volatility': round(volatility, 3),
        'average': round(average, 3)
    }


def get_emotional_profile(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build emotional profile from responses.
    """
    if not responses:
        return {
            'dominant_emotion': 'neutral',
            'emotion_distribution': {},
            'emotional_range': 0.0
        }
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    all_scores = []
    
    for resp in responses:
        score = resp.get('sentiment_score', 0)
        all_scores.append(score)
        
        if score > 0.2:
            positive_count += 1
        elif score < -0.2:
            negative_count += 1
        else:
            neutral_count += 1
    
    total = len(responses)
    distribution = {
        'positive': round(positive_count / total, 2),
        'negative': round(negative_count / total, 2),
        'neutral': round(neutral_count / total, 2)
    }
    
    # Determine dominant
    if positive_count > negative_count and positive_count > neutral_count:
        dominant = 'positive'
    elif negative_count > positive_count and negative_count > neutral_count:
        dominant = 'negative'
    else:
        dominant = 'neutral'
    
    # Calculate range
    if all_scores:
        emotional_range = max(all_scores) - min(all_scores)
    else:
        emotional_range = 0.0
    
    return {
        'dominant_emotion': dominant,
        'emotion_distribution': distribution,
        'emotional_range': round(emotional_range, 2)
    }
