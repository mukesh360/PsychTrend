"""
K-Means clustering module for behavioral pattern grouping
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter

from ml_engine.sentiment_context import (
    analyze_sentiment_context,
    get_blocked_archetypes,
    get_preferred_archetypes
)


# Predefined behavioral archetypes
BEHAVIORAL_ARCHETYPES = {
    'achiever': {
        'traits': ['goal-oriented', 'persistent', 'growth-minded', 'competitive'],
        'keywords': ['achieve', 'goal', 'success', 'accomplish', 'win', 'best', 'first'],
        'description': 'Driven by goals and measurable achievements',
        'requires_evidence': True  # STRICT: Cannot be assigned without explicit evidence
    },
    'explorer': {
        'traits': ['curious', 'adventurous', 'open-minded', 'creative'],
        'keywords': ['new', 'learn', 'discover', 'try', 'experience', 'different', 'creative'],
        'description': 'Motivated by new experiences and learning'
    },
    'connector': {
        'traits': ['collaborative', 'empathetic', 'supportive', 'relationship-focused'],
        'keywords': ['team', 'together', 'help', 'support', 'people', 'friend', 'family'],
        'description': 'Values relationships and team collaboration'
    },
    'stabilizer': {
        'traits': ['consistent', 'reliable', 'methodical', 'structured'],
        'keywords': ['routine', 'consistent', 'steady', 'regular', 'maintain', 'organize'],
        'description': 'Prefers stability and established routines'
    },
    'adapter': {
        'traits': ['flexible', 'resilient', 'pragmatic', 'resourceful'],
        'keywords': ['adapt', 'change', 'flexible', 'adjust', 'overcome', 'handle'],
        'description': 'Thrives in changing environments'
    },
    'innovator': {
        'traits': ['creative', 'visionary', 'independent', 'problem-solver'],
        'keywords': ['create', 'build', 'design', 'innovate', 'idea', 'solution', 'improve'],
        'description': 'Focuses on creating and improving',
        'requires_evidence': True  # STRICT: Cannot be assigned without explicit evidence
    },
    # NEW: Neutral/development-focused archetypes for negative sentiment
    'developing': {
        'traits': ['growing', 'learning', 'building', 'progressing'],
        'keywords': ['trying', 'working on', 'getting better', 'learning'],
        'description': 'Currently in a developmental phase, building skills and direction',
        'is_neutral': True
    },
    'exploring': {
        'traits': ['searching', 'questioning', 'uncertain', 'open'],
        'keywords': ['not sure', 'figuring out', 'exploring', 'considering'],
        'description': 'Currently exploring options and directions',
        'is_neutral': True
    },
    'emerging': {
        'traits': ['transitioning', 'evolving', 'adapting', 'changing'],
        'keywords': ['changing', 'transition', 'shift', 'moving'],
        'description': 'In transition, with patterns still forming',
        'is_neutral': True
    },
    'uncertain': {
        'traits': ['questioning', 'reflective', 'undecided', 'contemplative'],
        'keywords': ['unsure', 'confused', 'unclear', 'questioning'],
        'description': 'Currently facing uncertainty about direction or goals',
        'is_neutral': True
    }
}


def extract_behavioral_features(responses: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract numerical features from responses for clustering.
    
    Features:
    - Average sentiment
    - Sentiment volatility
    - Growth keyword density
    - Challenge keyword density
    - Social keyword density
    - Achievement keyword density
    """
    if not responses:
        return np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    # Keyword categories
    growth_kw = {'learn', 'grow', 'improve', 'develop', 'progress', 'better'}
    challenge_kw = {'challenge', 'difficult', 'hard', 'struggle', 'overcome', 'face'}
    social_kw = {'team', 'people', 'together', 'help', 'family', 'friend', 'support'}
    achievement_kw = {'achieve', 'success', 'accomplish', 'win', 'complete', 'goal'}
    
    # Calculate features
    sentiments = [r.get('sentiment_score', 0) for r in responses]
    avg_sentiment = np.mean(sentiments)
    sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0
    
    # Keyword densities
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    word_count = len(all_text.split()) or 1
    
    growth_count = sum(1 for kw in growth_kw if kw in all_text)
    challenge_count = sum(1 for kw in challenge_kw if kw in all_text)
    social_count = sum(1 for kw in social_kw if kw in all_text)
    achievement_count = sum(1 for kw in achievement_kw if kw in all_text)
    
    features = np.array([[
        (avg_sentiment + 1) / 2,  # Normalize to 0-1
        min(1, sentiment_volatility),
        min(1, growth_count / 5),
        min(1, challenge_count / 5),
        min(1, social_count / 5),
        min(1, achievement_count / 5)
    ]])
    
    return features


def calculate_archetype_affinity(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Calculate affinity scores for each behavioral archetype.
    Uses sentiment context to block inappropriate archetypes and prefer neutral ones.
    
    STRICT RULES:
    - Block 'Achiever' unless explicit achievement evidence found
    - Block 'Innovator' unless explicit creative evidence found
    - Prefer neutral archetypes (developing, exploring, emerging) for negative sentiment
    """
    if not responses:
        return [{'cluster_name': 'unknown', 'affinity': 0.5, 'traits': [], 'description': 'Insufficient data'}]
    
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Get blocked and preferred archetypes based on sentiment
    blocked_archetypes = get_blocked_archetypes(sentiment_context)
    preferred_archetypes = get_preferred_archetypes(sentiment_context)
    is_negative_dominant = sentiment_context.get('is_negative_dominant', False)
    
    # Combine all text
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    all_keywords = []
    for r in responses:
        all_keywords.extend(r.get('keywords', []))
    
    keyword_counts = Counter(all_keywords)
    
    affinities = []
    
    for archetype, data in BEHAVIORAL_ARCHETYPES.items():
        # STRICT: Skip blocked archetypes entirely
        if archetype in blocked_archetypes:
            continue
        
        score = 0.0
        matches = 0
        
        # Check keyword matches
        for kw in data['keywords']:
            if kw in all_text:
                matches += 1
                score += 0.12
        
        # Check extracted keyword matches
        for kw in keyword_counts:
            if any(trait in kw.lower() for trait in data['traits']):
                score += 0.08 * keyword_counts[kw]
        
        # STRICT: Require evidence for evidence-based archetypes
        if data.get('requires_evidence', False) and matches < 2:
            continue  # Skip if not enough evidence
        
        # Boost preferred archetypes for negative sentiment
        if archetype in preferred_archetypes:
            score += 0.3  # Significant boost
        
        # Penalty for positive archetypes in negative sentiment context
        if is_negative_dominant and not data.get('is_neutral', False):
            score = score * 0.6
        
        # Normalize score
        affinity = min(1.0, score)
        
        if affinity > 0.15:  # Lower threshold to include neutral archetypes
            affinities.append({
                'cluster_name': archetype,
                'affinity': round(affinity, 2),
                'traits': data['traits'],
                'description': data['description'],
                'is_neutral': data.get('is_neutral', False)
            })
    
    # Sort by affinity
    affinities.sort(key=lambda x: x['affinity'], reverse=True)
    
    # Return top 3 or at least one
    if not affinities:
        # STRICT: Default to neutral profile for negative sentiment
        if is_negative_dominant:
            return [{
                'cluster_name': 'developing',
                'affinity': 0.5,
                'traits': ['growing', 'learning', 'building', 'progressing'],
                'description': 'Currently in a developmental phase, building skills and direction',
                'is_neutral': True
            }]
        else:
            return [{
                'cluster_name': 'balanced',
                'affinity': 0.5,
                'traits': ['adaptable', 'moderate', 'flexible'],
                'description': 'Shows balanced behavioral patterns across categories'
            }]
    
    return affinities[:3]


def simple_kmeans(features: np.ndarray, k: int = 3, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple K-Means implementation (for demonstration without sklearn dependency issues).
    """
    n_samples = features.shape[0]
    
    if n_samples < k:
        k = n_samples
    
    # Initialize centroids randomly
    np.random.seed(42)
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = features[centroid_indices].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iter):
        # Assign labels
        for i in range(n_samples):
            distances = np.sum((centroids - features[i]) ** 2, axis=1)
            labels[i] = np.argmin(distances)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = features[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids


def cluster_responses_by_category(responses: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """
    Group responses by their category and analyze patterns within each.
    """
    categories = {}
    
    for resp in responses:
        cat = resp.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(resp)
    
    category_analysis = {}
    for cat, resps in categories.items():
        sentiments = [r.get('sentiment_score', 0) for r in resps]
        keywords = []
        for r in resps:
            keywords.extend(r.get('keywords', []))
        
        category_analysis[cat] = {
            'response_count': len(resps),
            'avg_sentiment': round(np.mean(sentiments), 2) if sentiments else 0,
            'top_keywords': [k for k, _ in Counter(keywords).most_common(3)]
        }
    
    return category_analysis


def get_behavioral_clusters(responses: List[Dict[str, Any]], sentiment_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main clustering function returning behavioral profile.
    Uses sentiment context for strict archetype assignment.
    """
    # Get sentiment context if not provided
    if sentiment_context is None:
        sentiment_context = analyze_sentiment_context(responses)
    
    # Calculate archetype affinities with sentiment awareness
    archetypes = calculate_archetype_affinity(responses, sentiment_context)
    
    # Get category analysis
    category_analysis = cluster_responses_by_category(responses)
    
    # Extract features for potential future ML
    features = extract_behavioral_features(responses)
    
    return {
        'archetypes': archetypes,
        'category_analysis': category_analysis,
        'feature_vector': features.tolist()[0] if len(features) > 0 else [],
        'sentiment_context': sentiment_context  # Include for downstream use
    }
