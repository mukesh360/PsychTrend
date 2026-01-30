# Psychological Trend Analysis System

A chatbot-based behavioral analysis system that collects personal experience data, analyzes behavioral trends, and generates non-clinical insights and predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)

## Features

- **Conversational Interface**: Chat-style UI for natural data collection
- **Adaptive Questioning**: Dynamic follow-up questions based on responses
- **ML-Powered Analysis**: Sentiment analysis, trend detection, and clustering
- **Behavioral Predictions**: Probability-based predictions with explanations
- **Comprehensive Reports**: Trend visualization, strengths, and growth areas
- **Dark/Light Theme**: User-selectable themes with responsive design
- **Local Data Storage**: SQLite database for privacy-first data handling

## Tech Stack

### Backend
- Python 3.9+
- FastAPI
- Pydantic
- Pandas, NumPy
- scikit-learn
- SQLite

### Frontend
- HTML5
- CSS3
- Bootstrap 5
- Vanilla JavaScript
- Font Awesome / Bootstrap Icons

## Project Structure

```
HACKTHON/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic data models
│   ├── database.py          # SQLite operations
│   ├── chat_logic.py        # Adaptive questioning
│   ├── data_processor.py    # Response processing
│   └── ml_engine/
│       ├── __init__.py
│       ├── sentiment.py     # Sentiment analysis
│       ├── trends.py        # Trend detection
│       ├── clustering.py    # K-Means clustering
│       └── predictor.py     # Behavior prediction
├── frontend/
│   ├── index.html           # Chat interface
│   ├── report.html          # Report display
│   ├── css/
│   │   └── styles.css       # Custom styles
│   └── js/
│       ├── chat.js          # Chat functionality
│       └── report.js        # Report rendering
├── data/
│   ├── questions.json       # Question bank
│   ├── sample_data.json     # Example dataset
│   └── psych_analysis.db    # SQLite database (auto-created)
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone or Navigate to Project

```bash
cd c:\HACKTHON
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open in Browser

Navigate to: **http://localhost:8000**

## Usage

1. **Start Conversation**: Click "Start Conversation" on the welcome screen
2. **Answer Questions**: Respond naturally to the chatbot's questions
3. **Complete Interview**: The bot covers education, career, milestones, habits, and challenges
4. **View Report**: Once complete, click "View Report" for your analysis

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve main chat page |
| POST | `/session` | Create new session |
| POST | `/chat` | Send/receive messages |
| GET | `/session/{id}` | Get session info |
| GET | `/analysis/{id}` | Get raw analysis data |
| GET | `/report/{id}` | Get formatted report |
| DELETE | `/session/{id}` | Delete session |
| POST | `/reset` | Delete all data |
| GET | `/health` | Health check |

## Analysis Features

### Trend Analysis
- **Motivation Trend**: Tracks enthusiasm and goal orientation
- **Consistency Score**: Measures behavioral stability
- **Growth Orientation**: Assesses learning mindset
- **Stress Response**: Analyzes coping patterns

### Behavioral Clustering
- Achiever, Explorer, Connector, Stabilizer, Adapter, Innovator archetypes
- Affinity scores for each archetype

### Predictions
- Consistency likelihood
- Adaptability to change
- Growth potential
- Attention areas (non-clinical)

## Ethical Considerations

⚠️ **This system is NOT a medical or psychological diagnostic tool.**

- All outputs use non-clinical language
- Clear disclaimers on every report
- User data stored locally only
- Full data deletion available

## Sample Conversation

```
Bot: Hello! I'm here to help you reflect on your life experiences. 
     What should I call you?
User: Alex

Bot: Nice to meet you, Alex! Let's talk about your educational journey. 
     What was your most memorable learning experience?
User: Completing my engineering degree was tough but rewarding. 
      I struggled initially but found my passion in programming.

Bot: That sounds like a great experience! What made it special?
User: The challenge of solving complex problems and seeing my code work.

[...continues through career, milestones, habits, challenges...]

Bot: Thank you for sharing! Your insight report is ready.
```

## Example Report Output

```json
{
  "executive_summary": "Based on 8 responses from Alex, the analysis reveals 
    a primarily 'Achiever' behavioral profile. Driven by goals and measurable 
    achievements. Overall sentiment tendency is 0.65 (positive).",
  "trend_analysis": {
    "motivation": {"score": 0.72, "direction": "upward"},
    "consistency": {"score": 0.68, "direction": "stable"},
    "growth_orientation": {"score": 0.85, "direction": "upward"}
  },
  "strengths": [
    "Goal-oriented with strong achievement drive",
    "Natural inclination toward personal growth",
    "Demonstrated resilience in facing challenges"
  ],
  "growth_opportunities": [
    "Developing stress management techniques",
    "Improving work-life balance"
  ]
}
```

## Contributing

This project is designed as a learning prototype suitable for:
- College final-year projects
- Hackathon submissions
- Resume portfolio pieces

## License

MIT License - Feel free to use and modify for educational purposes.

---

**Disclaimer**: This system provides behavioral insights for self-reflection only. 
It is not a substitute for professional psychological or medical advice.

---

## 🤖 LOCAL LLM INTEGRATION (Ollama)

This system integrates **Ollama** with the **qwen2.5:7b** model to enhance answer quality, insight explanations, and report generation while maintaining strict ethical guardrails.

### Architecture Overview

```
User Input
    ↓
┌─────────────────────────────────────┐
│  LLM Layer 1: Input Normalization   │  ← Clean weak inputs ("kk", "idk")
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  ML Analysis Layer (Source of Truth)│  ← Sentiment, Trends, Clustering
│  - Sentiment Analysis               │
│  - Trend Detection                  │
│  - Behavioral Clustering            │
│  - Predictions                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LLM Layer 2: Report Generation     │  ← Humanize insights & write reports
└─────────────────────────────────────┘
    ↓
Final Report
```

> ⚠️ **IMPORTANT**: The LLM NEVER bypasses the ML analysis layer. All statistical outputs remain the source of truth.

### LLM Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Input Normalization** | Cleans weak inputs like "kk", "idk", "meh" | Better feature extraction |
| **Question Enhancement** | Generates engaging follow-up questions | Improved conversation flow |
| **Insight Explanation** | Converts numerical trends to human language | Understandable reports |
| **Report Generation** | Creates comprehensive behavioral reports | Executive summaries |

### Installation

#### 1. Install Ollama

```powershell
# Windows (using winget)
winget install Ollama.Ollama

# Or download from: https://ollama.ai/download
```

#### 2. Pull the qwen2.5:7b Model

```bash
ollama pull qwen2.5:7b
```

#### 3. Verify Installation

```bash
# Check Ollama is running
ollama list

# Should show: qwen2.5:7b
```

#### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### New API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/llm/health` | Check Ollama LLM connection status |
| GET | `/report-enhanced/{session_id}` | Get LLM-enhanced comprehensive report |

### Example: LLM Health Check

```bash
curl http://localhost:8000/llm/health
```

Response:
```json
{
  "status": "healthy",
  "ollama_running": true,
  "model_available": true,
  "configured_model": "qwen2.5:7b",
  "available_models": ["qwen2.5:7b"]
}
```

### Example: Input Normalization

**Before (weak input):**
```
"kk yeah I guess work is fine nothing special"
```

**After (normalized):**
```
"Work has been stable, without notable events."
```

### Example: LLM-Enhanced Report

```bash
curl http://localhost:8000/report-enhanced/{session_id}
```

Response includes:
```json
{
  "executive_summary": "Based on 12 responses from Alex, you demonstrate a primarily 'Achiever' behavioral profile with strong goal orientation (0.72) and consistent growth mindset (0.85). Your responses indicate positive engagement across career and learning domains. Note: This is not a medical or psychological diagnosis.",
  "full_report_markdown": "## Your Behavioral Profile\n...",
  "trend_explanations": {
    "motivation": "Your motivation score of 0.72 shows an upward trend, indicating increasing enthusiasm...",
    "consistency": "With a consistency score of 0.68, you demonstrate reliable behavioral patterns..."
  },
  "llm_enhanced": true,
  "llm_model": "qwen2.5:7b"
}
```

### Guardrails & Safety

The LLM integration includes multiple safety layers:

1. **Forbidden Terms Validation**
   - Blocks clinical/diagnostic language
   - Sanitizes output if violations detected

2. **Prompt Engineering**
   - System prompts enforce non-clinical language
   - Explicit rules prevent hallucination

3. **Fallback Mechanisms**
   - Graceful degradation if Ollama is unavailable
   - Template-based fallbacks for all features

4. **Output Validation**
   - All LLM outputs checked before display
   - Automatic sanitization of problematic content

### LLM Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Model | `qwen2.5:7b` | Balance of quality and speed |
| Temperature | `0.2` | Consistent, factual outputs |
| Max Tokens | `1024` | Sufficient for report sections |
| Timeout | `60s` | Handle slower inference |

### File Structure (LLM Components)

```
backend/
├── ollama_client.py    # Async HTTP client for Ollama API
├── llm_prompts.py      # Prompt templates with guardrails
├── llm_service.py      # High-level LLM service functions
└── ...existing files...
```

### For Hackathon Judges

**Key Innovation Points:**

1. **Local LLM** - Privacy-respecting, runs entirely on local hardware
2. **Layered Architecture** - ML analysis is source of truth, LLM only explains
3. **Ethical Guardrails** - Prevents clinical language and hallucination
4. **Graceful Fallback** - System works without LLM if unavailable
5. **Production-Ready** - Retry logic, health checks, error handling

**Test the Integration:**

```bash
# 1. Start Ollama (if not running)
ollama serve

# 2. Start the backend
cd backend
uvicorn main:app --reload --port 8000

# 3. Check LLM health
curl http://localhost:8000/llm/health

# 4. Complete a chat session and get enhanced report
curl http://localhost:8000/report-enhanced/{session_id}
```



Implementation setup ::
# Psychological Trend Analysis System - Implementation Plan

Build a complete chatbot-based behavioral analysis system with adaptive questioning, ML-powered trend detection, and insight report generation.

---

## Proposed Changes

### Project Structure

```
c:\HACKTHON\
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── database.py          # SQLite setup
│   ├── chat_logic.py        # Adaptive questioning
│   ├── data_processor.py    # Response structuring
│   └── ml_engine/
│       ├── __init__.py
│       ├── sentiment.py     # Sentiment analysis
│       ├── trends.py        # Time-series analysis
│       ├── clustering.py    # K-Means clustering
│       └── predictor.py     # Behavior prediction
├── frontend/
│   ├── index.html           # Main chat interface
│   ├── report.html          # Report display page
│   ├── css/
│   │   └── styles.css       # Custom styles
│   └── js/
│       ├── chat.js          # Chat functionality
│       └── report.js        # Report rendering
├── data/
│   ├── questions.json       # Question bank
│   └── sample_data.json     # Example dataset
├── requirements.txt
└── README.md
```

---

### Backend Components

#### [NEW] [main.py](file:///c:/HACKTHON/backend/main.py)
- FastAPI app with CORS middleware
- Endpoints: `/chat`, `/session`, `/analysis`, `/report`, `/reset`
- Serve static frontend files

#### [NEW] [models.py](file:///c:/HACKTHON/backend/models.py)
- `Message`: user/bot message structure
- `ChatSession`: session tracking
- `StructuredResponse`: processed user data
- `PredictionResult`: analysis output

#### [NEW] [database.py](file:///c:/HACKTHON/backend/database.py)
- SQLite connection for local storage
- Tables: `sessions`, `responses`, `reports`
- CRUD operations

#### [NEW] [chat_logic.py](file:///c:/HACKTHON/backend/chat_logic.py)
- Question categories: education, career, milestones, habits, challenges
- Context-aware follow-up selection
- Conversation flow management

#### [NEW] [data_processor.py](file:///c:/HACKTHON/backend/data_processor.py)
- Convert chat to structured JSON
- Extract: category, event_description, timestamp, sentiment_score
- Handle incomplete responses

---

### ML Analysis Engine

#### [NEW] [sentiment.py](file:///c:/HACKTHON/backend/ml_engine/sentiment.py)
- Rule-based sentiment scoring using word lists
- Polarity calculation (-1 to +1 scale)

#### [NEW] [trends.py](file:///c:/HACKTHON/backend/ml_engine/trends.py)
- Time-series trend detection
- Moving averages for motivation/consistency
- Change point detection

#### [NEW] [clustering.py](file:///c:/HACKTHON/backend/ml_engine/clustering.py)
- K-Means for behavioral pattern grouping
- Feature extraction from responses

#### [NEW] [predictor.py](file:///c:/HACKTHON/backend/ml_engine/predictor.py)
- Random Forest for behavior prediction
- Probability-based outputs
- Explainable feature importance

---

### Frontend Components

#### [NEW] [index.html](file:///c:/HACKTHON/frontend/index.html)
- Bootstrap 5 chat interface
- Message bubbles (user/bot differentiation)
- Typing indicator animation
- Dark/light theme toggle
- Responsive layout

#### [NEW] [report.html](file:///c:/HACKTHON/frontend/report.html)
- Trend visualizations
- Behavioral summaries
- Strength patterns & growth areas
- Disclaimer prominently displayed

#### [NEW] [styles.css](file:///c:/HACKTHON/frontend/css/styles.css)
- Chat bubble styling
- Animations (typing, message entry)
- Theme variables
- Responsive breakpoints

#### [NEW] [chat.js](file:///c:/HACKTHON/frontend/js/chat.js)
- Fetch API calls to backend
- Dynamic message rendering
- Auto-scroll behavior
- Session management

---

### Data Files

#### [NEW] [questions.json](file:///c:/HACKTHON/data/questions.json)
- Question bank organized by category
- Follow-up question mappings
- Conversational phrasing

#### [NEW] [sample_data.json](file:///c:/HACKTHON/data/sample_data.json)
- Example user responses
- Demo analysis results

---

## Verification Plan

### Automated Tests
```bash
# Install dependencies
cd c:\HACKTHON
pip install -r requirements.txt

# Run backend server
cd backend
uvicorn main:app --reload --port 8000

# API endpoint tests (using curl or Postman)
# 1. Create session: POST /session
# 2. Send message: POST /chat
# 3. Get analysis: GET /analysis/{session_id}
# 4. Get report: GET /report/{session_id}
```

### Manual Verification
1. **Chat Flow Test**: Open `http://localhost:8000` in browser, complete full conversation
2. **Theme Toggle**: Verify dark/light mode switching works
3. **Report Generation**: Complete chat and verify report displays with trends
4. **Reset Function**: Test data deletion works properly
5. **Responsive Check**: Test on mobile viewport sizes

---

## Ethical Safeguards

- All analysis text uses non-clinical language
- Disclaimer appears on every report:
  > "This is not medical or psychological diagnosis. For professional guidance, consult a qualified professional."
- User data stored locally only (SQLite)
- Complete data reset available via `/reset` endpoint

---

> [!IMPORTANT]
> This system provides behavioral insights only. No diagnostic or medical terminology will be used in any output.
