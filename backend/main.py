"""
FastAPI Main Application - Psychological Trend Analysis System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime
from typing import Optional

from models import (
    ChatRequest, ChatResponse, SessionCreate, SessionResponse,
    ResetResponse, AnalysisResult, ReportResponse,
    LLMHealthStatus, EnhancedReportResponse
)
import database as db
from chat_logic import get_next_question, start_conversation
from data_processor import structure_response, process_incomplete_response, aggregate_session_data
from ml_engine.sentiment import analyze_sentiment_detailed, get_emotional_profile
from ml_engine.trends import get_all_trends
from ml_engine.clustering import get_behavioral_clusters
from ml_engine.predictor import get_predictions, identify_strengths, identify_growth_areas

# LLM Integration
from llm_service import get_llm_service
from ollama_client import check_ollama_health, DEFAULT_MODEL


# Initialize FastAPI app
app = FastAPI(
    title="Psychological Trend Analysis System",
    description="A chatbot-based behavioral analysis system for non-clinical insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files path
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"


# Mount static files if frontend exists
if FRONTEND_PATH.exists():
    # Mount CSS and JS directories at their expected paths
    css_path = FRONTEND_PATH / "css"
    js_path = FRONTEND_PATH / "js"
    
    if css_path.exists():
        app.mount("/css", StaticFiles(directory=str(css_path)), name="css")
    if js_path.exists():
        app.mount("/js", StaticFiles(directory=str(js_path)), name="js")


# ============== ROUTES ==============

@app.get("/")
async def root():
    """Serve main page"""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Psychological Trend Analysis System API", "status": "running"}


@app.get("/index.html")
async def index_html():
    """Serve main page (explicit path)"""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Psychological Trend Analysis System API", "status": "running"}


@app.get("/report.html")
async def report_html():
    """Serve report page"""
    report_path = FRONTEND_PATH / "report.html"
    if report_path.exists():
        return FileResponse(str(report_path))
    raise HTTPException(status_code=404, detail="Report page not found")


@app.get("/report-page")
async def report_page():
    """Serve report page (alternative route)"""
    report_path = FRONTEND_PATH / "report.html"
    if report_path.exists():
        return FileResponse(str(report_path))
    raise HTTPException(status_code=404, detail="Report page not found")


@app.post("/session", response_model=SessionResponse)
async def create_session(session_data: Optional[SessionCreate] = None):
    """Create a new chat session"""
    user_name = session_data.user_name if session_data else None
    session_id = db.create_session(user_name)
    
    # Get first message
    first_message, category, progress = start_conversation(session_id)
    db.add_to_conversation(session_id, 'bot', first_message)
    
    return SessionResponse(
        session_id=session_id,
        created_at=datetime.now(),
        message=first_message
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat message with LLM-enhanced natural responses"""
    session = db.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Process user message
    is_valid, processed = process_incomplete_response(request.message)
    
    if not is_valid:
        return ChatResponse(
            session_id=request.session_id,
            message=processed,
            is_complete=False,
            current_category=session.get('current_category', ''),
            progress=0.0
        )
    
    # Save user message
    db.add_to_conversation(request.session_id, 'user', request.message)
    
    # Structure and save response data
    current_category = session.get('current_category', 'introduction')
    if current_category not in ['introduction', 'closing']:
        structured = structure_response(
            request.message,
            current_category,
            request.session_id
        )
        db.add_response(request.session_id, structured)
    
    # Get next question - use LLM-enhanced version for natural responses
    try:
        from chat_logic import get_next_question_enhanced
        next_message, is_complete, category, progress = await get_next_question_enhanced(
            request.session_id,
            request.message
        )
    except Exception:
        # Fallback to standard logic if LLM fails
        next_message, is_complete, category, progress = get_next_question(
            request.session_id,
            request.message
        )
    
    # Save bot message
    db.add_to_conversation(request.session_id, 'bot', next_message)
    
    return ChatResponse(
        session_id=request.session_id,
        message=next_message,
        is_complete=is_complete,
        current_category=category,
        progress=progress
    )


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = db.get_conversation_history(session_id)
    
    return {
        "session_id": session_id,
        "user_name": session.get('user_name'),
        "is_complete": bool(session.get('is_complete', 0)),
        "current_category": session.get('current_category', ''),
        "conversation_history": history
    }


@app.get("/analysis/{session_id}")
async def get_analysis(session_id: str):
    """Get behavioral analysis for a session"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    responses = db.get_session_responses(session_id)
    
    if len(responses) < 3:
        return {
            "status": "insufficient_data",
            "message": "Need more responses for meaningful analysis",
            "response_count": len(responses)
        }
    
    # Perform analyses
    aggregated = aggregate_session_data(responses)
    trends = get_all_trends(responses)
    clusters = get_behavioral_clusters(responses)
    predictions = get_predictions(responses)
    emotional_profile = get_emotional_profile(responses)
    strengths = identify_strengths(responses)
    growth_areas = identify_growth_areas(responses)
    
    return {
        "session_id": session_id,
        "user_name": session.get('user_name'),
        "analysis_timestamp": datetime.now().isoformat(),
        "response_count": len(responses),
        "aggregated_data": aggregated,
        "trends": trends,
        "behavioral_clusters": clusters,
        "predictions": predictions,
        "emotional_profile": emotional_profile,
        "strengths": strengths,
        "growth_areas": growth_areas
    }


@app.get("/report/{session_id}", response_model=ReportResponse)
async def get_report(session_id: str):
    """Generate comprehensive report"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    responses = db.get_session_responses(session_id)
    
    # Get all analyses
    aggregated = aggregate_session_data(responses)
    trends = get_all_trends(responses)
    clusters = get_behavioral_clusters(responses)
    predictions = get_predictions(responses)
    strengths = identify_strengths(responses)
    growth_areas = identify_growth_areas(responses)
    
    # Build executive summary
    user_name = session.get('user_name', 'User')
    dominant_archetype = clusters['archetypes'][0] if clusters['archetypes'] else None
    
    if dominant_archetype:
        archetype_name = dominant_archetype['cluster_name'].title()
        summary = (
            f"Based on {len(responses)} responses from {user_name}, the analysis reveals "
            f"a primarily '{archetype_name}' behavioral profile. "
            f"{dominant_archetype['description']}. "
            f"Overall sentiment tendency is {aggregated['overall_sentiment']:.2f} "
            f"({'positive' if aggregated['overall_sentiment'] > 0 else 'negative' if aggregated['overall_sentiment'] < 0 else 'neutral'})."
        )
    else:
        summary = (
            f"Analysis of {len(responses)} responses from {user_name} shows "
            f"a balanced behavioral profile with varied patterns across categories."
        )
    
    # Format trend analysis
    trend_analysis = {
        'motivation': {
            'score': trends['motivation']['score'],
            'direction': trends['motivation']['trend_direction'],
            'description': trends['motivation']['description']
        },
        'consistency': {
            'score': trends['consistency']['score'],
            'direction': trends['consistency']['trend_direction'],
            'description': trends['consistency']['description']
        },
        'growth_orientation': {
            'score': trends['growth']['score'],
            'direction': trends['growth']['trend_direction'],
            'description': trends['growth']['description']
        },
        'stress_response': {
            'pattern': trends['stress_response'].get('pattern', 'balanced'),
            'score': trends['stress_response']['score'],
            'description': trends['stress_response']['description']
        }
    }
    
    # Format behavioral profile
    behavioral_profile = {
        'primary_archetype': clusters['archetypes'][0] if clusters['archetypes'] else None,
        'secondary_archetypes': clusters['archetypes'][1:] if len(clusters['archetypes']) > 1 else [],
        'category_breakdown': clusters['category_analysis']
    }
    
    # Format predictions
    formatted_predictions = [
        {
            'type': p['prediction_type'],
            'probability': p['probability'] if 'probability' in p else None,
            'confidence': p['confidence'],
            'explanation': p['explanation'],
            'factors': p['contributing_factors']
        }
        for p in predictions
    ]
    
    # Save report
    report_data = {
        'executive_summary': summary,
        'trend_analysis': trend_analysis,
        'behavioral_profile': behavioral_profile,
        'predictions': formatted_predictions,
        'strengths': strengths,
        'growth_opportunities': growth_areas
    }
    db.save_report(session_id, report_data)
    
    return ReportResponse(
        session_id=session_id,
        generated_at=datetime.now(),
        user_name=user_name,
        executive_summary=summary,
        trend_analysis=trend_analysis,
        behavioral_profile=behavioral_profile,
        predictions=formatted_predictions,
        strengths=strengths,
        growth_opportunities=growth_areas
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    deleted = db.delete_session(session_id)
    if deleted:
        return {"success": True, "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/reset", response_model=ResetResponse)
async def reset_all_data():
    """Reset all data - delete all sessions and responses"""
    count = db.delete_all_data()
    return ResetResponse(
        success=True,
        message="All data has been deleted",
        deleted_sessions=count
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/llm/health", response_model=LLMHealthStatus)
async def llm_health_check():
    """Check Ollama LLM connection status"""
    health = await check_ollama_health()
    return LLMHealthStatus(
        status=health.get("status", "error"),
        ollama_running=health.get("ollama_running", False),
        model_available=health.get("model_available", False),
        configured_model=health.get("configured_model", DEFAULT_MODEL),
        available_models=health.get("available_models", []),
        error=health.get("error")
    )


@app.get("/report-enhanced/{session_id}", response_model=EnhancedReportResponse)
async def get_enhanced_report(session_id: str):
    """
    Generate LLM-enhanced comprehensive report.
    Falls back to standard report if LLM is unavailable.
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    responses = db.get_session_responses(session_id)
    
    if len(responses) < 3:
        raise HTTPException(
            status_code=400,
            detail="Need at least 3 responses for report generation"
        )
    
    # Get all ML analyses (source of truth)
    aggregated = aggregate_session_data(responses)
    trends = get_all_trends(responses)
    clusters = get_behavioral_clusters(responses)
    predictions = get_predictions(responses)
    strengths = identify_strengths(responses)
    growth_areas = identify_growth_areas(responses)
    
    user_name = session.get('user_name', 'User')
    
    # Get LLM service for enhanced generation
    llm_service = get_llm_service()
    
    # Generate LLM-enhanced report
    llm_report = await llm_service.generate_full_report(
        user_name=user_name,
        response_count=len(responses),
        trends=trends,
        clusters=clusters,
        predictions=predictions,
        strengths=strengths,
        growth_areas=growth_areas
    )
    
    # Format trend analysis for response
    trend_analysis = {
        'motivation': {
            'score': trends['motivation']['score'],
            'direction': trends['motivation']['trend_direction'],
            'description': trends['motivation']['description']
        },
        'consistency': {
            'score': trends['consistency']['score'],
            'direction': trends['consistency']['trend_direction'],
            'description': trends['consistency']['description']
        },
        'growth_orientation': {
            'score': trends['growth']['score'],
            'direction': trends['growth']['trend_direction'],
            'description': trends['growth']['description']
        },
        'stress_response': {
            'pattern': trends['stress_response'].get('pattern', 'balanced'),
            'score': trends['stress_response']['score'],
            'description': trends['stress_response']['description']
        }
    }
    
    # Format behavioral profile
    behavioral_profile = {
        'primary_archetype': clusters['archetypes'][0] if clusters['archetypes'] else None,
        'secondary_archetypes': clusters['archetypes'][1:] if len(clusters['archetypes']) > 1 else [],
        'category_breakdown': clusters['category_analysis']
    }
    
    # Format predictions
    formatted_predictions = [
        {
            'type': p['prediction_type'],
            'probability': p.get('probability'),
            'confidence': p['confidence'],
            'explanation': p['explanation'],
            'factors': p['contributing_factors']
        }
        for p in predictions
    ]
    
    return EnhancedReportResponse(
        session_id=session_id,
        generated_at=datetime.now(),
        user_name=user_name,
        executive_summary=llm_report.get('executive_summary', ''),
        full_report_markdown=llm_report.get('full_report_markdown'),
        trend_analysis=trend_analysis,
        trend_explanations=llm_report.get('trend_explanations', {}),
        behavioral_profile=behavioral_profile,
        predictions=formatted_predictions,
        strengths=llm_report.get('strengths', strengths),
        growth_opportunities=llm_report.get('growth_opportunities', growth_areas),
        llm_enhanced=llm_report.get('llm_enhanced', False),
        llm_model=DEFAULT_MODEL if llm_report.get('llm_enhanced') else None
    )


# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
