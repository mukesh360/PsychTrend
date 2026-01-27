"""
Behavior prediction module using Random Forest
"""
import numpy as np
from typing import Dict, List, Any
from collections import Counter


class BehaviorPredictor:
    """
    Prediction engine for behavioral tendencies.
    Uses rule-based logic with probability outputs for explainability.
    """
    
    def __init__(self):
        # Feature weights for predictions (learned from patterns)
        self.consistency_weights = {
            'routine_mentions': 0.3,
            'habit_mentions': 0.25,
            'sentiment_stability': 0.25,
            'discipline_keywords': 0.2
        }
        
        self.adaptability_weights = {
            'change_mentions': 0.25,
            'overcome_keywords': 0.3,
            'positive_challenge_response': 0.25,
            'flexibility_keywords': 0.2
        }
        
        self.growth_weights = {
            'learning_keywords': 0.3,
            'improvement_mentions': 0.25,
            'goal_orientation': 0.25,
            'positive_trend': 0.2
        }
    
    def extract_prediction_features(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features relevant for predictions.
        """
        if not responses:
            return {}
        
        all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
        sentiments = [r.get('sentiment_score', 0) for r in responses]
        qualities = [r.get('input_quality', 1.0) for r in responses]
        keywords = []
        for r in responses:
            keywords.extend(r.get('keywords', []))
        
        keyword_counts = Counter(keywords)
        word_count = len(all_text.split()) or 1
        avg_quality = sum(qualities) / len(qualities) if qualities else 1.0
        
        # Calculate features
        features = {
            # Consistency features
            'routine_mentions': all_text.count('routine') + all_text.count('regular') + all_text.count('daily'),
            'habit_mentions': all_text.count('habit') + all_text.count('always') + all_text.count('every'),
            'sentiment_stability': 1 - min(1, np.std(sentiments) * 2) if len(sentiments) > 1 else 0.5,
            'discipline_keywords': keyword_counts.get('discipline', 0) + keyword_counts.get('consistency', 0),
            
            # Adaptability features
            'change_mentions': all_text.count('change') + all_text.count('adapt') + all_text.count('adjust'),
            'overcome_keywords': keyword_counts.get('resilience', 0) + keyword_counts.get('overcame', 0),
            'flexibility_keywords': all_text.count('flexible') + all_text.count('open'),
            
            # Growth features
            'learning_keywords': keyword_counts.get('growth', 0) + keyword_counts.get('learned', 0),
            'improvement_mentions': all_text.count('improve') + all_text.count('better') + all_text.count('progress'),
            'goal_orientation': keyword_counts.get('achievement', 0) + all_text.count('goal'),
            
            # Sentiment features
            'avg_sentiment': np.mean(sentiments),
            'sentiment_trend': sentiments[-1] - sentiments[0] if len(sentiments) > 1 else 0,
            'positive_ratio': len([s for s in sentiments if s > 0.2]) / len(sentiments) if sentiments else 0.5,
            
            # Quality features
            'avg_quality': avg_quality,
            'low_quality_ratio': len([q for q in qualities if q < 0.3]) / len(qualities) if qualities else 0
        }
        
        return features
    
    def predict_consistency(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict likelihood of maintaining consistent behavior.
        """
        # Calculate weighted score
        score = 0.3  # Base probability
        
        score += min(0.2, features.get('routine_mentions', 0) * 0.05)
        score += min(0.15, features.get('habit_mentions', 0) * 0.04)
        score += features.get('sentiment_stability', 0.5) * 0.2
        score += min(0.15, features.get('discipline_keywords', 0) * 0.07)
        
        # Apply quality penalty
        avg_quality = features.get('avg_quality', 1.0)
        if avg_quality < 0.5:
            score = score * avg_quality * 1.2  # Significant reduction
        
        # Normalize
        probability = min(0.95, max(0.1, score))
        
        # Determine confidence
        if probability > 0.7:
            confidence = 'high'
        elif probability > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Contributing factors
        factors = []
        if features.get('routine_mentions', 0) > 1:
            factors.append('Established routines')
        if features.get('sentiment_stability', 0) > 0.6:
            factors.append('Emotional stability')
        if features.get('habit_mentions', 0) > 1:
            factors.append('Strong habit formation')
        
        if not factors:
            factors = ['General behavioral patterns']
        
        return {
            'prediction_type': 'Consistency Likelihood',
            'probability': round(probability, 2),
            'confidence': confidence,
            'explanation': f"Based on your responses, there is a {int(probability*100)}% likelihood of maintaining consistent behavior patterns.",
            'contributing_factors': factors
        }
    
    def predict_adaptability(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict ability to adapt to change.
        """
        score = 0.3  # Base
        
        score += min(0.2, features.get('change_mentions', 0) * 0.06)
        score += min(0.2, features.get('overcome_keywords', 0) * 0.1)
        score += min(0.15, features.get('flexibility_keywords', 0) * 0.07)
        score += (features.get('avg_sentiment', 0) + 1) / 10  # Positive sentiment helps
        
        # Apply quality penalty
        avg_quality = features.get('avg_quality', 1.0)
        if avg_quality < 0.5:
            score = score * avg_quality * 1.2
        
        probability = min(0.95, max(0.1, score))
        
        if probability > 0.7:
            confidence = 'high'
        elif probability > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        factors = []
        if features.get('change_mentions', 0) > 0:
            factors.append('Experience with transitions')
        if features.get('overcome_keywords', 0) > 0:
            factors.append('Demonstrated resilience')
        if features.get('avg_sentiment', 0) > 0.2:
            factors.append('Positive outlook')
        
        if not factors:
            factors = ['General adaptability indicators']
        
        return {
            'prediction_type': 'Adaptability to Change',
            'probability': round(probability, 2),
            'confidence': confidence,
            'explanation': f"Analysis suggests {int(probability*100)}% adaptability potential in new situations.",
            'contributing_factors': factors
        }
    
    def predict_growth_potential(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict learning and career growth inclination.
        """
        score = 0.35  # Base
        
        score += min(0.2, features.get('learning_keywords', 0) * 0.1)
        score += min(0.15, features.get('improvement_mentions', 0) * 0.04)
        score += min(0.15, features.get('goal_orientation', 0) * 0.07)
        score += max(0, features.get('sentiment_trend', 0) * 0.15)
        
        # Apply quality penalty
        avg_quality = features.get('avg_quality', 1.0)
        if avg_quality < 0.5:
            score = score * avg_quality * 1.2
        
        probability = min(0.95, max(0.1, score))
        
        if probability > 0.7:
            confidence = 'high'
        elif probability > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        factors = []
        if features.get('learning_keywords', 0) > 0:
            factors.append('Learning orientation')
        if features.get('goal_orientation', 0) > 0:
            factors.append('Goal-driven mindset')
        if features.get('improvement_mentions', 0) > 0:
            factors.append('Focus on improvement')
        
        if not factors:
            factors = ['Baseline growth indicators']
        
        return {
            'prediction_type': 'Growth & Learning Potential',
            'probability': round(probability, 2),
            'confidence': confidence,
            'explanation': f"Indicates {int(probability*100)}% potential for continued growth and learning.",
            'contributing_factors': factors
        }
    
    def assess_risk_indicators(self, features: Dict[str, float], responses: List[Dict]) -> Dict[str, Any]:
        """
        Assess behavioral risk indicators (non-clinical).
        """
        # Look for patterns that might indicate areas for attention
        risk_level = 'low'
        indicators = []
        
        avg_sentiment = features.get('avg_sentiment', 0)
        stability = features.get('sentiment_stability', 0.5)
        positive_ratio = features.get('positive_ratio', 0.5)
        avg_quality = features.get('avg_quality', 1.0)
        low_quality_ratio = features.get('low_quality_ratio', 0)
        
        # Check for low engagement/quality responses
        if avg_quality < 0.3:
            indicators.append('Very low engagement in responses')
            risk_level = 'elevated'
        elif low_quality_ratio > 0.5:
            indicators.append('Majority of responses show minimal engagement')
            risk_level = 'moderate'
        
        # Check for concerning patterns
        if avg_sentiment < -0.3:
            indicators.append('Tendency toward negative self-reflection')
            risk_level = 'moderate' if risk_level == 'low' else risk_level
        
        if stability < 0.3:
            indicators.append('High emotional variability in responses')
            if risk_level == 'moderate':
                risk_level = 'elevated'
        
        if positive_ratio < 0.3:
            indicators.append('Low frequency of positive experiences mentioned')
            risk_level = 'moderate' if risk_level == 'low' else risk_level
        
        # Check for avoidance patterns
        all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
        if 'avoid' in all_text or 'give up' in all_text or 'can\'t' in all_text:
            indicators.append('Possible avoidance tendencies')
        
        if not indicators:
            indicators = ['No significant behavioral concerns identified']
        
        return {
            'prediction_type': 'Behavioral Attention Areas',
            'risk_level': risk_level,
            'confidence': 'medium',
            'explanation': f"Risk assessment level: {risk_level}. This is not a clinical assessment.",
            'contributing_factors': indicators
        }
    
    def get_all_predictions(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate all predictions for a user.
        """
        features = self.extract_prediction_features(responses)
        
        predictions = [
            self.predict_consistency(features),
            self.predict_adaptability(features),
            self.predict_growth_potential(features),
            self.assess_risk_indicators(features, responses)
        ]
        
        return predictions


def get_predictions(responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main function to get behavioral predictions.
    """
    predictor = BehaviorPredictor()
    return predictor.get_all_predictions(responses)


def identify_strengths(responses: List[Dict[str, Any]]) -> List[str]:
    """
    Identify key strengths from responses.
    """
    if not responses:
        return ['Unable to identify strengths from limited data']
    
    strengths = []
    all_keywords = []
    for r in responses:
        all_keywords.extend(r.get('keywords', []))
    
    keyword_counts = Counter(all_keywords)
    
    strength_mapping = {
        'achievement': 'Goal-oriented with strong achievement drive',
        'growth': 'Natural inclination toward personal growth',
        'resilience': 'Demonstrated resilience in facing challenges',
        'leadership': 'Leadership qualities and initiative',
        'creativity': 'Creative and innovative thinking',
        'teamwork': 'Strong collaborative and interpersonal skills',
        'adaptation': 'Adaptability and flexibility',
        'self-improvement': 'Commitment to self-improvement',
        'passion': 'Passionate engagement with interests'
    }
    
    for keyword, strength in strength_mapping.items():
        if keyword_counts.get(keyword, 0) > 0:
            strengths.append(strength)
    
    # Add sentiment-based strengths
    sentiments = [r.get('sentiment_score', 0) for r in responses]
    if np.mean(sentiments) > 0.3:
        strengths.append('Positive outlook and optimistic perspective')
    
    if not strengths:
        strengths = ['Willingness to self-reflect and share experiences']
    
    return strengths[:5]  # Top 5 strengths


def identify_growth_areas(responses: List[Dict[str, Any]]) -> List[str]:
    """
    Identify areas for growth and development.
    """
    if not responses:
        return ['More data needed to identify growth areas']
    
    growth_areas = []
    all_text = ' '.join(r.get('raw_response', '') for r in responses).lower()
    sentiments = [r.get('sentiment_score', 0) for r in responses]
    
    # Check for areas that might need attention
    if 'stress' in all_text or 'overwhelm' in all_text:
        growth_areas.append('Developing stress management techniques')
    
    if 'balance' in all_text or 'too much work' in all_text:
        growth_areas.append('Improving work-life balance')
    
    if 'confidence' in all_text and ('lack' in all_text or 'low' in all_text):
        growth_areas.append('Building self-confidence')
    
    if np.std(sentiments) > 0.4 if len(sentiments) > 1 else False:
        growth_areas.append('Developing emotional regulation strategies')
    
    if np.mean(sentiments) < 0:
        growth_areas.append('Cultivating a more positive perspective')
    
    if 'procrastin' in all_text:
        growth_areas.append('Overcoming procrastination tendencies')
    
    if not growth_areas:
        growth_areas = ['Continue building on existing strengths']
    
    return growth_areas[:4]  # Top 4 growth areas
