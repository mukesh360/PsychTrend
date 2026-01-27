"""
Pydantic models for the Psychological Trend Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    BOT = "bot"


class SentimentCategory(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Message(BaseModel):
    """Single chat message"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Incoming chat message from user"""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Bot response to user"""
    session_id: str
    message: str
    is_complete: bool = False
    current_category: Optional[str] = None
    progress: float = 0.0


class StructuredResponse(BaseModel):
    """Processed user response"""
    category: str
    event_description: str
    timestamp: datetime
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_category: SentimentCategory
    keywords: List[str] = []
    raw_response: str


class SessionCreate(BaseModel):
    """Create new session"""
    user_name: Optional[str] = None


class SessionResponse(BaseModel):
    """Session creation response"""
    session_id: str
    created_at: datetime
    message: str


class TrendData(BaseModel):
    """Trend analysis result"""
    name: str
    score: float
    trend_direction: str  # "upward", "downward", "stable"
    description: str
    data_points: List[float] = []


class ClusterResult(BaseModel):
    """Clustering analysis result"""
    cluster_name: str
    affinity: float
    traits: List[str]
    description: str


class PredictionResult(BaseModel):
    """Behavior prediction"""
    prediction_type: str
    probability: float
    confidence: str  # "high", "medium", "low"
    explanation: str
    contributing_factors: List[str]


class AnalysisResult(BaseModel):
    """Complete analysis output"""
    session_id: str
    user_name: Optional[str]
    analysis_timestamp: datetime
    total_responses: int
    
    # Trend metrics
    motivation_trend: TrendData
    consistency_score: TrendData
    growth_orientation: TrendData
    stress_response: TrendData
    
    # Clustering
    behavioral_clusters: List[ClusterResult]
    
    # Predictions
    predictions: List[PredictionResult]
    
    # Summary
    strengths: List[str]
    growth_areas: List[str]
    overall_summary: str


class ReportResponse(BaseModel):
    """Final report for display"""
    session_id: str
    generated_at: datetime
    user_name: Optional[str]
    
    # Main sections
    executive_summary: str
    trend_analysis: Dict[str, Any]
    behavioral_profile: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    strengths: List[str]
    growth_opportunities: List[str]
    
    # Disclaimer
    disclaimer: str = (
        "IMPORTANT DISCLAIMER: This report provides behavioral insights based on "
        "self-reported experiences. It is NOT a medical, clinical, or psychological "
        "diagnosis. For professional guidance on mental health or personal development, "
        "please consult a qualified professional."
    )


class ResetResponse(BaseModel):
    """Data reset confirmation"""
    success: bool
    message: str
    deleted_sessions: int = 0
