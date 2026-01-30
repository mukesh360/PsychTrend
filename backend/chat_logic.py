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
        
        elif questions_in_category == 2 and user_response:
            # Check if user said no/not ready
            response_lower = user_response.strip().lower()
            negative_responses = {'no', 'nope', 'nah', 'not ready', 'not yet', 'later', 'no thanks'}
            
            if response_lower in negative_responses or response_lower.startswith('no'):
                # User said no - acknowledge and ask again gently
                session = db.get_session(session_id)
                name = session.get('user_name', 'Friend')
                return (
                    f"No problem, {name}! Take your time. Just let me know when you're ready to begin by saying 'yes' or 'ready'. 😊",
                    False, current_category, progress
                )
            
            # User said yes or anything else - proceed
            # Move to next category
            return move_to_next_category(session_id, category_index, total_categories)
        
        # Move to next category (fallback)
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


# =============================================================================
# LLM-Enhanced Conversational Responses
# =============================================================================

async def humanize_bot_response(
    base_message: str,
    user_name: str,
    user_response: Optional[str] = None,
    category: str = ""
) -> str:
    """
    Use LLM to make bot responses more natural and human-like.
    Falls back to base_message if LLM is unavailable.
    """
    try:
        from llm_service import get_llm_service
        
        llm_service = get_llm_service()
        
        if not await llm_service.is_available():
            return base_message
        
        # Create a prompt for humanizing the response
        prompt = f"""Rewrite this chatbot message to sound more warm, natural, and human-like.

ORIGINAL MESSAGE: "{base_message}"
USER'S NAME: {user_name}
CONTEXT: Conversation about {category if category else 'getting to know them'}
{f'USER JUST SAID: "{user_response}"' if user_response else ''}

RULES:
- Keep the same meaning and intent
- Make it sound like a friendly human, not a robot
- Use natural conversational language
- Add appropriate warmth and empathy
- Keep it concise (1-2 sentences max)
- Do NOT add clinical or medical language
- Do NOT ask multiple questions

Return ONLY the rewritten message, nothing else."""

        result = await llm_service.client.generate(
            prompt=prompt,
            system_prompt="You are a warm, empathetic conversation partner helping someone reflect on their life experiences. Be natural and human-like.",
            temperature=0.4,
            max_tokens=150
        )
        
        if result.get("success"):
            humanized = result.get("response", "").strip()
            # Remove any quotes if the LLM added them
            humanized = humanized.strip('"\'')
            if humanized and len(humanized) > 10:
                return humanized
        
        return base_message
    
    except Exception:
        return base_message


async def generate_empathetic_response(
    user_response: str,
    category: str,
    next_question: str,
    user_name: str
) -> str:
    """
    Generate an empathetic acknowledgment + next question using LLM.
    Makes the conversation feel more natural and responsive.
    """
    try:
        from llm_service import get_llm_service
        
        llm_service = get_llm_service()
        
        if not await llm_service.is_available():
            return next_question
        
        prompt = f"""You are having a friendly conversation with {user_name} about their life experiences.

They just shared: "{user_response}"
Topic: {category}
Next question to ask: "{next_question}"

Generate a natural response that:
1. Briefly acknowledges what they shared (1 short sentence)
2. Then asks the next question naturally

RULES:
- Be warm, supportive, and empathetic
- Sound like a caring friend, not a therapist or robot
- Keep total response to 2-3 sentences max
- Do NOT use clinical or psychological terms
- Do NOT analyze or diagnose
- Make it feel like natural conversation

Return ONLY the response, no explanations."""

        result = await llm_service.client.generate(
            prompt=prompt,
            system_prompt="You are a warm, empathetic friend helping someone reflect on their life journey. Be genuine and caring.",
            temperature=0.5,
            max_tokens=200
        )
        
        if result.get("success"):
            response = result.get("response", "").strip()
            response = response.strip('"\'')
            
            # Validate no clinical terms
            from llm_prompts import validate_output, sanitize_output
            is_valid, _ = validate_output(response)
            if not is_valid:
                response = sanitize_output(response)
            
            if response and len(response) > 20:
                return response
        
        return next_question
    
    except Exception:
        return next_question


async def get_next_question_enhanced(
    session_id: str,
    user_response: Optional[str] = None
) -> Tuple[str, bool, str, float]:
    """
    Enhanced version of get_next_question that uses LLM for natural responses.
    """
    # Get base response from standard logic
    base_message, is_complete, category, progress = get_next_question(session_id, user_response)
    
    # If complete or no response to react to, just humanize the message
    if is_complete or not user_response:
        session = db.get_session(session_id)
        user_name = session.get('user_name', 'Friend') if session else 'Friend'
        humanized = await humanize_bot_response(base_message, user_name, user_response, category)
        return (humanized, is_complete, category, progress)
    
    # For ongoing conversation, generate empathetic response
    session = db.get_session(session_id)
    user_name = session.get('user_name', 'Friend') if session else 'Friend'
    
    # Skip humanization for introduction phase (already handled)
    if category == "introduction":
        return (base_message, is_complete, category, progress)
    
    # Generate empathetic response with LLM
    enhanced_response = await generate_empathetic_response(
        user_response=user_response,
        category=category,
        next_question=base_message,
        user_name=user_name
    )
    
    return (enhanced_response, is_complete, category, progress)

