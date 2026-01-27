"""
Data processing module - converts chat responses to structured data
"""
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json


# Sentiment word lists
POSITIVE_WORDS = {
    'happy', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
    'enjoy', 'success', 'successful', 'achieved', 'proud', 'excited', 'passionate',
    'motivated', 'inspired', 'accomplished', 'grateful', 'thankful', 'blessed',
    'confident', 'strong', 'resilient', 'determined', 'focused', 'creative',
    'innovative', 'learned', 'grew', 'improved', 'overcame', 'won', 'best',
    'better', 'positive', 'optimistic', 'hopeful', 'energetic', 'productive'
}

NEGATIVE_WORDS = {
    'sad', 'difficult', 'hard', 'challenging', 'struggled', 'failed', 'failure',
    'stressed', 'anxious', 'worried', 'frustrated', 'disappointed', 'unhappy',
    'confused', 'lost', 'stuck', 'overwhelmed', 'tired', 'exhausted', 'burned',
    'rejected', 'afraid', 'scared', 'nervous', 'doubt', 'uncertain', 'weak',
    'lonely', 'isolated', 'hurt', 'pain', 'problem', 'issue', 'mistake', 'regret'
}

INTENSITY_MODIFIERS = {
    'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
    'somewhat': 0.5, 'slightly': 0.5, 'a bit': 0.5, 'kind of': 0.5,
    'absolutely': 2.0, 'totally': 1.5, 'completely': 2.0
}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nobody', 'nothing'}

# Keyword extraction patterns
KEYWORD_PATTERNS = {
    'achievement': r'\b(achieved|accomplished|succeeded|won|completed|finished|earned)\b',
    'growth': r'\b(grew|learned|improved|developed|progressed|advanced|evolved)\b',
    'challenge': r'\b(struggled|overcame|faced|dealt with|handled|managed|survived)\b',
    'passion': r'\b(love|passionate|enjoy|excited|interested|fascinated)\b',
    'resilience': r'\b(resilient|persistent|determined|persevered|bounced back)\b',
    'leadership': r'\b(led|managed|directed|organized|coordinated|mentored)\b',
    'creativity': r'\b(created|designed|innovated|invented|built|developed)\b',
    'teamwork': r'\b(team|collaborated|together|group|helped|supported)\b',
    'self-improvement': r'\b(self-taught|practice|routine|habit|discipline)\b',
    'adaptation': r'\b(adapted|adjusted|changed|flexible|transitioned)\b'
}


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of text using rule-based approach.
    Returns score from -1.0 (negative) to 1.0 (positive).
    """
    if not text:
        return 0.0
    
    words = text.lower().split()
    score = 0.0
    word_count = 0
    
    i = 0
    while i < len(words):
        word = re.sub(r'[^\w]', '', words[i])
        
        # Check for negation
        is_negated = False
        if i > 0:
            prev_word = re.sub(r'[^\w]', '', words[i-1])
            if prev_word in NEGATION_WORDS or "'t" in words[i-1]:
                is_negated = True
        
        # Check for intensity modifier
        modifier = 1.0
        if i > 0:
            prev_word = words[i-1].lower()
            for mod, value in INTENSITY_MODIFIERS.items():
                if mod in prev_word:
                    modifier = value
                    break
        
        # Calculate word score
        if word in POSITIVE_WORDS:
            word_score = 1.0 * modifier
            if is_negated:
                word_score = -word_score * 0.5
            score += word_score
            word_count += 1
        elif word in NEGATIVE_WORDS:
            word_score = -1.0 * modifier
            if is_negated:
                word_score = -word_score * 0.5
            score += word_score
            word_count += 1
        
        i += 1
    
    # Normalize score
    if word_count > 0:
        normalized_score = score / word_count
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized_score))
    
    return 0.0


def get_sentiment_category(score: float) -> str:
    """Convert sentiment score to category"""
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    return "neutral"


def extract_keywords(text: str) -> List[str]:
    """Extract relevant keywords from text"""
    keywords = []
    text_lower = text.lower()
    
    for keyword, pattern in KEYWORD_PATTERNS.items():
        if re.search(pattern, text_lower):
            keywords.append(keyword)
    
    return keywords


def extract_event_description(text: str, category: str) -> str:
    """Create a concise event description from raw response"""
    # Clean and truncate text
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # If text is short enough, use as is
    if len(cleaned) <= 200:
        return cleaned
    
    # Otherwise, create summary (first 200 chars + "...")
    return cleaned[:197] + "..."


def structure_response(
    raw_response: str,
    category: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Convert raw chat response into structured data.
    
    Returns dict with:
    - category
    - event_description
    - timestamp
    - sentiment_score
    - sentiment_category
    - keywords
    - raw_response
    """
    sentiment_score = analyze_sentiment(raw_response)
    
    structured = {
        'session_id': session_id,
        'category': category,
        'event_description': extract_event_description(raw_response, category),
        'timestamp': datetime.now().isoformat(),
        'sentiment_score': round(sentiment_score, 3),
        'sentiment_category': get_sentiment_category(sentiment_score),
        'keywords': extract_keywords(raw_response),
        'raw_response': raw_response
    }
    
    return structured


def process_incomplete_response(response: str) -> Tuple[bool, str]:
    """
    Handle incomplete or unclear responses.
    Returns (is_valid, processed_response or error message)
    """
    if not response or len(response.strip()) < 2:
        return (False, "I didn't catch that. Could you please share a bit more?")
    
    # Check for very short non-answers
    short_responses = {'ok', 'yes', 'no', 'maybe', 'idk', 'sure', 'fine', 'good'}
    if response.strip().lower() in short_responses:
        return (True, response)  # Accept but won't provide rich data
    
    return (True, response.strip())


def aggregate_session_data(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate all responses from a session for analysis.
    """
    if not responses:
        return {
            'total_responses': 0,
            'categories': {},
            'overall_sentiment': 0.0,
            'all_keywords': [],
            'sentiment_timeline': []
        }
    
    # Group by category
    categories = {}
    all_keywords = []
    sentiment_timeline = []
    
    for resp in responses:
        cat = resp.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = {
                'responses': [],
                'avg_sentiment': 0.0,
                'keywords': []
            }
        
        categories[cat]['responses'].append(resp)
        categories[cat]['keywords'].extend(resp.get('keywords', []))
        
        all_keywords.extend(resp.get('keywords', []))
        sentiment_timeline.append({
            'timestamp': resp.get('timestamp'),
            'score': resp.get('sentiment_score', 0.0),
            'category': cat
        })
    
    # Calculate category averages
    for cat_data in categories.values():
        scores = [r.get('sentiment_score', 0) for r in cat_data['responses']]
        cat_data['avg_sentiment'] = sum(scores) / len(scores) if scores else 0.0
        cat_data['keywords'] = list(set(cat_data['keywords']))
    
    # Overall sentiment
    all_scores = [r.get('sentiment_score', 0) for r in responses]
    overall_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return {
        'total_responses': len(responses),
        'categories': categories,
        'overall_sentiment': round(overall_sentiment, 3),
        'all_keywords': list(set(all_keywords)),
        'sentiment_timeline': sentiment_timeline
    }
