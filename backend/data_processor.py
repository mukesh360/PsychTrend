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

# Low-quality input patterns
DISMISSIVE_RESPONSES = {
    'no', 'nope', 'nah', 'nothing', 'none', 'idk', 'dunno', 'na', 'n/a',
    'ok', 'okay', 'k', 'kk', 'yes', 'yeah', 'yep', 'sure', 'fine', 'good',
    'maybe', 'whatever', 'meh', 'eh', 'um', 'uh', 'hmm', 'hm'
}

# Pattern for repeated characters (e.g., "kk", "jjj", "hhhh")
REPEATED_CHAR_PATTERN = re.compile(r'^(.)\1+$')

# Pattern for random characters (e.g., "jj4", "asdf", "qwer")
RANDOM_CHAR_PATTERN = re.compile(r'^[a-z0-9]{1,4}$', re.IGNORECASE)


def validate_input_quality(text: str) -> float:
    """
    Validate the quality of user input.
    Returns a score from 0.0 (very low quality) to 1.0 (high quality).
    """
    if not text:
        return 0.0
    
    cleaned = text.strip().lower()
    word_count = len(cleaned.split())
    char_count = len(cleaned)
    
    # Single character or very short meaningless input
    if char_count <= 2:
        return 0.1
    
    # Check for repeated characters (kk, jjj, hhh)
    if REPEATED_CHAR_PATTERN.match(cleaned):
        return 0.1
    
    # Check for random character sequences (jj4, asdf, qw)
    if RANDOM_CHAR_PATTERN.match(cleaned) and cleaned not in {'i', 'a', 'ok', 'no'}:
        return 0.15
    
    # Check for dismissive single-word responses
    if cleaned in DISMISSIVE_RESPONSES:
        return 0.2
    
    # Check for mixed dismissive + random (e.g., "no no", "kk ok")
    words = cleaned.split()
    dismissive_count = sum(1 for w in words if w in DISMISSIVE_RESPONSES)
    if word_count <= 3 and dismissive_count == word_count:
        return 0.2
    
    # Short responses (less than 10 chars, less than 3 words)
    if char_count < 10 and word_count < 3:
        return 0.3
    
    # Check if response has any meaningful content words
    meaningful_words = set(cleaned.split()) - DISMISSIVE_RESPONSES - {'the', 'a', 'an', 'is', 'was', 'i', 'my', 'me'}
    if not meaningful_words and word_count < 5:
        return 0.25
    
    # Moderate length but check for substance
    if word_count < 5:
        return 0.5
    
    # Decent response
    if word_count < 10:
        return 0.7
    
    # Good, detailed response
    return 1.0


def is_meaningful_response(text: str) -> bool:
    """Check if a response is meaningful enough for analysis."""
    return validate_input_quality(text) >= 0.3


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
    
    # No sentiment words found - check input quality
    # Low-quality inputs should return negative scores instead of neutral
    quality = validate_input_quality(text)
    if quality < 0.3:
        # Very low quality: return negative score proportional to how bad the quality is
        return -0.4 * (1 - quality / 0.3)  # Returns -0.4 to -0.13
    elif quality < 0.5:
        # Mediocre quality: slight negative
        return -0.1
    
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
    - input_quality (0.0 to 1.0)
    """
    sentiment_score = analyze_sentiment(raw_response)
    input_quality = validate_input_quality(raw_response)
    
    structured = {
        'session_id': session_id,
        'category': category,
        'event_description': extract_event_description(raw_response, category),
        'timestamp': datetime.now().isoformat(),
        'sentiment_score': round(sentiment_score, 3),
        'sentiment_category': get_sentiment_category(sentiment_score),
        'keywords': extract_keywords(raw_response),
        'raw_response': raw_response,
        'input_quality': round(input_quality, 2)
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
