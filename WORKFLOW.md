# PsychTrend - System Workflow & Architecture

A comprehensive guide to how PsychTrend processes user responses and generates behavioral insight reports.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture Diagram](#architecture-diagram)
4. [Data Flow](#data-flow)
5. [Core Components](#core-components)
6. [Scoring & Analysis Pipeline](#scoring--analysis-pipeline)
7. [LLM Integration (Ollama + Qwen)](#llm-integration-ollama--qwen)
8. [Local Data Storage](#local-data-storage)
9. [API Endpoints](#api-endpoints)

---

## System Overview

PsychTrend is a **behavioral insight chatbot** that:
1. Engages users in a conversational assessment
2. Analyzes responses using NLP and statistical methods
3. Generates personalized behavioral insight reports
4. Uses a local LLM (Qwen via Ollama) for enhanced interpretations

**Key Features:**
- Non-clinical, self-reflection focused
- Stricter scoring rules for negative sentiment
- Local LLM processing (no cloud API required)
- SQLite-based session storage

---

## Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Async web framework for REST API |
| **Uvicorn** | ASGI server |
| **Pydantic** | Data validation & models |
| **SQLite + aiosqlite** | Local async database |

### ML/NLP
| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **scikit-learn** | ML algorithms (clustering, predictions) |
| **NLTK** | Natural language processing |

### LLM Integration
| Component | Purpose |
|-----------|---------|
| **Ollama** | Local LLM runtime |
| **Qwen 2.5:7B** | Language model for text generation |
| **httpx** | Async HTTP client for Ollama API |
| **tenacity** | Retry logic for API calls |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML/CSS/JavaScript** | Chat interface |
| **Vanilla CSS** | Modern styling with glassmorphism |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                              │
│                        (HTML/CSS/JavaScript)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                               │
│                         (main.py - Port 8000)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Chat Logic  │  │    Data     │  │   ML/NLP    │  │    LLM      │   │
│  │             │  │  Processor  │  │   Engine    │  │  Service    │   │
│  │ • Questions │  │ • Sentiment │  │ • Trends    │  │ • Prompts   │   │
│  │ • Flow      │  │ • Keywords  │  │ • Clusters  │  │ • Reports   │   │
│  │ • Context   │  │ • Quality   │  │ • Predict   │  │ • Normalize │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │                                                    │
         ▼                                                    ▼
┌─────────────────────┐                        ┌─────────────────────────┐
│   SQLite Database   │                        │   OLLAMA (Local LLM)    │
│ (psych_analysis.db) │                        │   Qwen 2.5:7B Model     │
│                     │                        │   http://localhost:11434│
│ • Sessions          │                        │                         │
│ • Responses         │                        │ • Input Normalization   │
│ • Reports           │                        │ • Insight Explanation   │
└─────────────────────┘                        │ • Report Generation     │
                                               └─────────────────────────┘
```

---

## Data Flow

### Step-by-Step Process

```
1. USER INPUT          2. PROCESSING           3. ANALYSIS            4. REPORT
   ─────────────────────────────────────────────────────────────────────────────
   
   "I feel stressed     ┌─────────────┐       ┌─────────────┐       ┌─────────┐
    and unmotivated"    │ Validate    │       │ Sentiment   │       │ Trends  │
         │              │ Quality     │       │ Analysis    │       │ (4 dim) │
         ▼              │ (0.0-1.0)   │       │ (-1 to +1)  │       │         │
   ┌───────────┐        └──────┬──────┘       └──────┬──────┘       │ Motiv-  │
   │ structure │               │                     │              │ ation   │
   │ _response │               ▼                     ▼              │         │
   └───────────┘        ┌─────────────┐       ┌─────────────┐       │ Consis- │
         │              │ Keyword     │       │ Sentiment   │       │ tency   │
         │              │ Extraction  │       │ Context     │       │         │
         │              │ (negation-  │       │ Detection   │       │ Growth  │
         │              │  aware)     │       │             │       │         │
         │              └──────┬──────┘       └──────┬──────┘       │ Stress  │
         │                     │                     │              │ Resp.   │
         ▼                     ▼                     ▼              └────┬────┘
   ┌───────────┐        ┌─────────────┐       ┌─────────────┐           │
   │ Store in  │        │ Keywords    │       │ Score Caps  │           ▼
   │ Database  │        │ + Sentiment │       │ Applied     │     ┌─────────┐
   └───────────┘        │ combined    │       │ (if neg)    │     │Archetype│
                        └─────────────┘       └─────────────┘     │Matching │
                                                                   └────┬────┘
                                                                        │
                                                                        ▼
                                                                  ┌─────────┐
                                                                  │ LLM     │
                                                                  │ Report  │
                                                                  │ Gen     │
                                                                  └─────────┘
```

---

## Core Components

### 1. Chat Logic (`chat_logic.py`)
Manages the conversational flow with users.

```python
# Key Functions:
start_conversation(user_name)     # Initialize session
get_next_question(context)        # Adaptive question selection
```

**Question Categories:**
- Learning & Education
- Career & Professional
- Personal Achievements
- Daily Routines & Habits
- Challenges & Stress

---

### 2. Data Processor (`data_processor.py`)
Converts raw text into structured, analyzable data.

```python
# Key Functions:
structure_response(raw_text, category, session_id)
analyze_sentiment(text)           # Rule-based: -1.0 to +1.0
extract_keywords(text)            # Context-aware, negation-checked
validate_input_quality(text)      # Quality score: 0.0 to 1.0
```

**Sentiment Analysis:**
- Word lists: POSITIVE_WORDS, NEGATIVE_WORDS
- Intensity modifiers: "very", "extremely", etc.
- Negation handling: "not happy" → negative

**Keyword Extraction (Context-Aware):**
```python
# STRICT: Only extracts keywords when NOT negated
"nothing helped me"    → NO 'teamwork' keyword
"I helped my team"     → 'teamwork' keyword extracted
```

---

### 3. ML Engine (`ml_engine/`)

#### Sentiment Context (`sentiment_context.py`)
Detects overall negative sentiment dominance.

```python
# Key Functions:
analyze_sentiment_context(responses)  # Detects negative dominance
get_score_caps(context)               # Returns max scores per trend
get_blocked_archetypes(context)       # Archetypes to avoid
```

**Score Caps (when negative sentiment dominates):**
| Trend | Maximum Score |
|-------|---------------|
| Motivation | ≤ 45% |
| Consistency | ≤ 30% |
| Growth Orientation | ≤ 45% |
| Stress Response | ≤ 45% |

---

#### Trends Analysis (`trends.py`)
Calculates 4 behavioral trend scores.

```python
# Trend Functions:
analyze_motivation_trend(responses, sentiment_context)
analyze_consistency(responses, sentiment_context)
analyze_growth_orientation(responses, sentiment_context)
analyze_stress_response(responses, sentiment_context)
```

**Scoring Logic:**
1. Base analysis from keywords/patterns
2. Apply penalties for negative language
3. Apply score caps if sentiment is negative
4. Generate appropriate description

---

#### Behavioral Clustering (`clustering.py`)
Assigns behavioral archetypes.

**Available Archetypes:**
| Type | Archetypes |
|------|------------|
| **Achievement** | Achiever*, Innovator* |
| **Exploration** | Explorer, Adapter |
| **Social** | Connector |
| **Stability** | Stabilizer |
| **Neutral** | Developing, Exploring, Emerging, Uncertain |

*Requires explicit evidence to be assigned

```python
# STRICT Rules:
# - Block 'Achiever' unless achievement keywords found
# - Prefer neutral archetypes for negative sentiment
```

---

#### Predictions (`predictor.py`)
Generates behavioral predictions.

```python
# Predictions:
predict_consistency(features)       # Likelihood of consistent behavior
predict_adaptability(features)      # Adaptation potential
predict_growth_potential(features)  # Learning inclination
assess_risk_indicators(features)    # Behavioral attention areas
```

**Strengths Identification (Strict):**
```python
# ONLY includes strengths with actual evidence
# Returns "No clear strengths identified" for negative sentiment
```

---

## Scoring & Analysis Pipeline

### Complete Pipeline

```
Raw Response
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 1. INPUT VALIDATION                                        │
│    • Quality check (0.0 - 1.0)                             │
│    • Low quality (< 0.3) → Reduced weight                  │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 2. SENTIMENT ANALYSIS                                      │
│    • Word matching (positive/negative lists)               │
│    • Negation detection ("not happy" → negative)           │
│    • Intensity modifiers ("very sad" → stronger)           │
│    • Output: score from -1.0 to +1.0                       │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 3. KEYWORD EXTRACTION                                      │
│    • Pattern matching for behavioral keywords              │
│    • Negation-aware (skip if preceded by "no", "never")    │
│    • Categories: achievement, growth, teamwork, etc.       │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 4. SENTIMENT CONTEXT                                       │
│    • Aggregate all responses                               │
│    • Detect negative dominance                             │
│    • Calculate score caps                                  │
│    • Identify attention areas                              │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 5. TREND SCORING (with caps applied)                       │
│    • Motivation: keywords + sentiment penalties            │
│    • Consistency: routine patterns + exhaustion penalty    │
│    • Growth: learning keywords + uncertainty penalty       │
│    • Stress Response: coping patterns + fear penalty       │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 6. ARCHETYPE ASSIGNMENT                                    │
│    • Calculate affinity scores                             │
│    • Block inappropriate archetypes (strict rules)         │
│    • Prefer neutral archetypes for negative sentiment      │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ 7. LLM REPORT GENERATION                                   │
│    • Pass all data to Qwen model                           │
│    • Anti-optimism-bias prompts                            │
│    • Generate human-readable report                        │
└────────────────────────────────────────────────────────────┘
```

---

## LLM Integration (Ollama + Qwen)

### Setup Requirements

```bash
# Install Ollama
# Windows: Download from https://ollama.ai

# Pull the model
ollama pull qwen2.5:7b

# Verify running
ollama list
```

### How Ollama Helps

| Feature | How LLM Enhances It |
|---------|---------------------|
| **Input Normalization** | Converts weak inputs like "idk" to meaningful statements |
| **Insight Explanation** | Turns numerical scores into human-readable explanations |
| **Report Generation** | Creates comprehensive behavioral reports |
| **Question Enhancement** | Makes follow-up questions more contextual |

### Communication Flow

```
┌──────────────┐         HTTP POST          ┌──────────────────┐
│   Backend    │ ───────────────────────▶   │     Ollama       │
│ (llm_service)│         /api/generate      │ localhost:11434  │
│              │ ◀───────────────────────   │                  │
│              │      JSON Response         │   Qwen 2.5:7B    │
└──────────────┘                            └──────────────────┘
```

### Ollama Client (`ollama_client.py`)

```python
class OllamaClient:
    base_url = "http://localhost:11434"
    model = "qwen2.5:7b"
    
    async def generate(prompt, system_prompt, temperature, max_tokens):
        # Sends request to Ollama API
        # Returns generated text
```

### LLM Prompts (`llm_prompts.py`)

**Anti-Optimism-Bias Rules:**
```python
# STRICT RULES in prompts:
# 1. Do NOT apply optimism bias
# 2. If score < 0.45, reflect accurately
# 3. Do NOT interpret struggle as achievement
# 4. Do NOT assume resilience without evidence
```

---

## Local Data Storage

### SQLite Database (`psych_analysis.db`)

Located in: `data/psych_analysis.db`

### Tables

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_name TEXT,
    created_at TIMESTAMP,
    status TEXT,         -- active, completed
    question_count INTEGER
);

-- Responses table
CREATE TABLE responses (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    category TEXT,
    raw_response TEXT,
    sentiment_score REAL,
    sentiment_category TEXT,
    keywords TEXT,       -- JSON array
    input_quality REAL,
    timestamp TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Reports table (cached)
CREATE TABLE reports (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    report_json TEXT,
    generated_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### Data Flow to Storage

```
User Response
     │
     ▼
structure_response() ──▶ JSON Object ──▶ db.save_response()
                              │
                              ▼
                        ┌─────────────┐
                        │   SQLite    │
                        │  Database   │
                        └─────────────┘
```

---

## API Endpoints

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/start` | Start new session |
| `POST` | `/chat/message` | Send user message |
| `GET` | `/analysis/{session_id}` | Get full analysis |
| `GET` | `/report/{session_id}` | Get generated report |
| `GET` | `/health` | Server health check |
| `GET` | `/llm/health` | Ollama health check |

### Example API Call

```bash
# Start conversation
curl -X POST http://localhost:8000/chat/start \
  -H "Content-Type: application/json" \
  -d '{"user_name": "John"}'

# Send message
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "message": "I feel stressed about work"
  }'
```

---

## Running the Project

```bash
# 1. Start Ollama (in separate terminal)
ollama serve

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
cd backend
uvicorn main:app --reload --port 8000

# 4. Open browser
http://localhost:8000
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, all endpoints |
| `chat_logic.py` | Conversation flow |
| `data_processor.py` | Text → structured data |
| `ml_engine/sentiment_context.py` | Negative sentiment detection |
| `ml_engine/trends.py` | 4 behavioral trend scores |
| `ml_engine/clustering.py` | Archetype assignment |
| `ml_engine/predictor.py` | Predictions & strengths |
| `llm_service.py` | LLM integration layer |
| `llm_prompts.py` | All LLM prompts |
| `ollama_client.py` | HTTP client for Ollama |
| `database.py` | SQLite operations |

---

## Important Notes

1. **Non-Clinical**: This is for self-reflection only, not diagnosis
2. **Local Processing**: All LLM processing is local via Ollama
3. **Strict Scoring**: Negative responses get appropriately low scores
4. **Evidence-Based**: Strengths require actual keyword evidence
5. **No Optimism Bias**: LLM prompts prevent positive reframing

---

*Last updated: January 30, 2026*
