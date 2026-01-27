"""
Adaptive questioning logic for the chatbot - Optimized 25-question flow
"""
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
from datetime import datetime

import database as db
from data_processor import analyze_sentiment


# Load questions
QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "questions.json"


def load_questions() -> Dict:
    """Load questions from JSON file"""
    try:
        with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return get_default_questions()


def get_default_questions() -> Dict:
    """Default question set"""
    return {
        "categories": {
            "introduction": {"questions": ["Hello! What should I call you?", "Nice to meet you! Ready to begin?"]},
            "education": {"questions": ["What was your most memorable learning experience?", "How do you approach learning new things?", "What challenges did you face in education?"]},
            "career": {"questions": ["What drew you to your field?", "What achievement are you proud of?", "How do you handle work pressure?"]},
            "milestones": {"questions": ["What's a turning point in your life?", "What personal accomplishment means most?", "What goal are you working towards?"]},
            "habits": {"questions": ["Describe your typical day?", "What habits keep you productive?", "What do you do for well-being?"]},
            "challenges": {"questions": ["What challenge have you overcome?", "How do you handle setbacks?", "What motivates you in tough times?"]},
            "closing": {"questions": ["Anything else to share?", "Your report is ready!"]}
        },
        "conversation_flow": ["introduction", "education", "career", "milestones", "habits", "challenges", "closing"],
        "questions_per_category": 3
    }


# Load configuration
QUESTIONS_DATA = load_questions()
FLOW = QUESTIONS_DATA.get("conversation_flow", [])
QUESTIONS_PER_CATEGORY = QUESTIONS_DATA.get("questions_per_category", 3)


def get_category_questions(category: str) -> List[str]:
    """Get questions for a category"""
    cat_data = QUESTIONS_DATA.get("categories", {}).get(category, {})
    return cat_data.get("questions", [])


def get_follow_up(category: str, sentiment: str) -> Optional[str]:
    """Get follow-up question based on sentiment"""
    cat_data = QUESTIONS_DATA.get("categories", {}).get(category, {})
    follow_ups = cat_data.get("follow_ups", {})
    options = follow_ups.get(sentiment, follow_ups.get("neutral", []))
    return random.choice(options) if options else None


def get_asked_questions(session_id: str) -> Set[str]:
    """Get set of already asked questions for this session"""
    session = db.get_session(session_id)
    if not session:
        return set()
    asked_json = session.get('asked_questions', '[]')
    try:
        return set(json.loads(asked_json))
    except:
        return set()


def add_asked_question(session_id: str, question: str):
    """Track that a question was asked"""
    asked = get_asked_questions(session_id)
    asked.add(question)
    db.update_session(session_id, asked_questions=json.dumps(list(asked)))


def get_next_unasked_question(questions: List[str], asked: Set[str]) -> Optional[str]:
    """Get the next question that hasn't been asked yet"""
    for q in questions:
        if q not in asked:
            return q
    return None


def get_next_question(session_id: str, user_response: Optional[str] = None) -> Tuple[str, bool, str, float]:
    """
    Determine the next question based on conversation state.
    Returns: (question, is_complete, current_category, progress)
    """
    session = db.get_session(session_id)
    if not session:
        return ("Session not found. Please start a new conversation.", True, "", 1.0)
    
    category_index = session.get('category_index', 0)
    questions_in_category = session.get('questions_in_category', 0)
    current_category = session.get('current_category', 'introduction')
    asked_questions = get_asked_questions(session_id)
    
    # Calculate progress
    total_categories = len(FLOW)
    progress = category_index / total_categories if total_categories > 0 else 0
    
    # Check if conversation is complete
    if category_index >= len(FLOW):
        db.update_session(session_id, is_complete=1)
        return (
            "Thank you for sharing! Your personalized insight report is now ready. Click 'View Report' to see your analysis.",
            True, "complete", 1.0
        )
    
    current_category = FLOW[category_index]
    questions = get_category_questions(current_category)
    
    # === INTRODUCTION ===
    if current_category == "introduction":
        if questions_in_category == 0:
            # First question - ask for name
            question = questions[0] if questions else "Hello! What should I call you?"
            add_asked_question(session_id, question)
            db.update_session(session_id, questions_in_category=1)
            return (question, False, current_category, progress)
        
        elif questions_in_category == 1 and user_response:
            # Save name and ask ready question
            name = user_response.strip().split()[0] if user_response else "Friend"
            db.update_session(session_id, user_name=name)
            
            if len(questions) > 1:
                question = f"Nice to meet you, {name}! {questions[1]}"
                add_asked_question(session_id, questions[1])
                db.update_session(session_id, questions_in_category=2)
                return (question, False, current_category, progress)
        
        # Move to next category
        return move_to_next_category(session_id, category_index, total_categories)
    
    # === CLOSING ===
    if current_category == "closing":
        next_q = get_next_unasked_question(questions, asked_questions)
        if next_q:
            add_asked_question(session_id, next_q)
            db.update_session(session_id, questions_in_category=questions_in_category + 1)
            return (next_q, False, current_category, progress)
        
        db.update_session(session_id, is_complete=1, category_index=len(FLOW))
        return (
            "Thank you for sharing your experiences! Your personalized insight report is ready. Click 'View Report' to see your behavioral analysis.",
            True, "complete", 1.0
        )
    
    # === REGULAR CATEGORIES ===
    # Check if we should ask a follow-up (30% chance, not on first question)
    if questions_in_category > 0 and user_response and random.random() < 0.3:
        sentiment = get_sentiment_category(user_response)
        follow_up = get_follow_up(current_category, sentiment)
        if follow_up and follow_up not in asked_questions:
            add_asked_question(session_id, follow_up)
            db.update_session(session_id, questions_in_category=questions_in_category + 1)
            return (follow_up, False, current_category, progress)
    
    # Get next unasked question
    next_q = get_next_unasked_question(questions, asked_questions)
    
    # Check if we've asked enough questions or exhausted this category
    if next_q and questions_in_category < QUESTIONS_PER_CATEGORY:
        add_asked_question(session_id, next_q)
        db.update_session(session_id, questions_in_category=questions_in_category + 1)
        return (next_q, False, current_category, progress)
    
    # Move to next category
    return move_to_next_category(session_id, category_index, total_categories)


def move_to_next_category(session_id: str, current_index: int, total: int) -> Tuple[str, bool, str, float]:
    """Move to the next category and return its first question"""
    next_index = current_index + 1
    
    if next_index >= len(FLOW):
        db.update_session(session_id, is_complete=1, category_index=len(FLOW))
        return ("Thank you! Your insight report is ready.", True, "complete", 1.0)
    
    next_category = FLOW[next_index]
    questions = get_category_questions(next_category)
    asked = get_asked_questions(session_id)
    
    db.update_session(
        session_id,
        category_index=next_index,
        questions_in_category=0,
        current_category=next_category
    )
    
    # Get transition and first question
    transition = get_category_transition(next_category)
    next_q = get_next_unasked_question(questions, asked)
    
    if next_q:
        add_asked_question(session_id, next_q)
        db.update_session(session_id, questions_in_category=1)
        question = f"{transition} {next_q}" if transition else next_q
    else:
        question = transition if transition else "Let's continue."
    
    new_progress = next_index / total
    return (question, False, next_category, new_progress)


def get_sentiment_category(text: str) -> str:
    """Categorize sentiment of text"""
    score = analyze_sentiment(text)
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    return "neutral"


def get_category_transition(category: str) -> str:
    """Get transition phrase for category"""
    transitions = {
        "education": "Let's explore your learning journey.",
        "career": "Now, let's talk about your professional path.",
        "milestones": "I'd love to hear about your personal milestones.",
        "habits": "Let's discuss your daily patterns and routines.",
        "challenges": "Now let's talk about how you handle challenges.",
        "closing": "We're almost done!"
    }
    return transitions.get(category, "")


def start_conversation(session_id: str) -> Tuple[str, str, float]:
    """Start a new conversation"""
    questions = get_category_questions("introduction")
    first_q = questions[0] if questions else "Hello! What should I call you?"
    
    add_asked_question(session_id, first_q)
    db.update_session(
        session_id,
        questions_in_category=1,
        current_category="introduction",
        asked_questions=json.dumps([first_q])
    )
    
    return (first_q, "introduction", 0.0)
