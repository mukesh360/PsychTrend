"""
SQLite database setup and operations for the Psychological Trend Analysis System
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid


# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "psych_analysis.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_name TEXT,
            created_at TEXT,
            updated_at TEXT,
            current_category TEXT DEFAULT 'introduction',
            category_index INTEGER DEFAULT 0,
            question_index INTEGER DEFAULT 0,
            questions_in_category INTEGER DEFAULT 0,
            asked_questions TEXT DEFAULT '[]',
            is_complete INTEGER DEFAULT 0,
            conversation_history TEXT DEFAULT '[]'
        )
    """)
    
    # Structured responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            category TEXT,
            event_description TEXT,
            timestamp TEXT,
            sentiment_score REAL,
            sentiment_category TEXT,
            keywords TEXT,
            raw_response TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    # Analysis reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            generated_at TEXT,
            report_data TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    conn.commit()
    conn.close()


def create_session(user_name: Optional[str] = None) -> str:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO sessions (session_id, user_name, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        (session_id, user_name, now, now)
    )
    conn.commit()
    conn.close()
    
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def update_session(session_id: str, **kwargs):
    """Update session fields"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build update query
    fields = []
    values = []
    for key, value in kwargs.items():
        if key == 'conversation_history' and isinstance(value, list):
            value = json.dumps(value)
        fields.append(f"{key} = ?")
        values.append(value)
    
    values.append(datetime.now().isoformat())
    values.append(session_id)
    
    query = f"UPDATE sessions SET {', '.join(fields)}, updated_at = ? WHERE session_id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()


def add_response(session_id: str, response_data: Dict[str, Any]):
    """Add structured response to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    keywords = json.dumps(response_data.get('keywords', []))
    
    cursor.execute(
        """INSERT INTO responses 
           (session_id, category, event_description, timestamp, sentiment_score, 
            sentiment_category, keywords, raw_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            response_data.get('category', ''),
            response_data.get('event_description', ''),
            response_data.get('timestamp', datetime.now().isoformat()),
            response_data.get('sentiment_score', 0.0),
            response_data.get('sentiment_category', 'neutral'),
            keywords,
            response_data.get('raw_response', '')
        )
    )
    conn.commit()
    conn.close()


def get_session_responses(session_id: str) -> List[Dict[str, Any]]:
    """Get all responses for a session"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM responses WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    responses = []
    for row in rows:
        response = dict(row)
        response['keywords'] = json.loads(response.get('keywords', '[]'))
        responses.append(response)
    
    return responses


def save_report(session_id: str, report_data: Dict[str, Any]):
    """Save analysis report"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT OR REPLACE INTO reports (session_id, generated_at, report_data)
           VALUES (?, ?, ?)""",
        (session_id, datetime.now().isoformat(), json.dumps(report_data))
    )
    conn.commit()
    conn.close()


def get_report(session_id: str) -> Optional[Dict[str, Any]]:
    """Get saved report for session"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM reports WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        report = dict(row)
        report['report_data'] = json.loads(report.get('report_data', '{}'))
        return report
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session and all related data"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM responses WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted


def delete_all_data() -> int:
    """Delete all data from database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM sessions")
    count = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM responses")
    cursor.execute("DELETE FROM reports")
    cursor.execute("DELETE FROM sessions")
    
    conn.commit()
    conn.close()
    
    return count


def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session"""
    session = get_session(session_id)
    if session:
        history = session.get('conversation_history', '[]')
        if isinstance(history, str):
            return json.loads(history)
        return history
    return []


def add_to_conversation(session_id: str, role: str, content: str):
    """Add message to conversation history"""
    history = get_conversation_history(session_id)
    history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    update_session(session_id, conversation_history=history)


# Initialize database on import
init_database()
