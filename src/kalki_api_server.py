"""
KALKI Construction Copilot API Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FastAPI server exposing Construction Copilot as REST API
Enables web/mobile apps to access KALKI's intelligence remotely

ENDPOINTS:
- POST /api/chat                - Main chat interface
- POST /api/roadmap             - Generate construction roadmap
- POST /api/consensus           - Multi-agent validation
- POST /api/property            - Property intelligence gathering
- GET  /health                  - Health check
- POST /api/auth/login          - User authentication
- POST /api/auth/register       - User registration

DEPLOYMENT:
- uvicorn kalki_api_server:app --host 0.0.0.0 --port 8000
- or: python3 kalki_api_server.py
"""

import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jose.jwt as jwt
from passlib.context import CryptContext
import uvicorn

# Import KALKI systems
from modules.construction_copilot_enhanced import EnhancedConstructionCopilot
from modules.llm import get_llm_engine

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="KALKI Construction Copilot API",
    description="AI-powered construction guidance system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://*.vercel.app",   # Vercel deployments
        # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global KALKI instance (loaded once at startup)
copilot: Optional[EnhancedConstructionCopilot] = None

# In-memory user database (replace with real DB in production)
users_db: Dict[str, Dict[str, Any]] = {}

# In-memory session store (replace with Redis in production)
active_sessions: Dict[str, Dict[str, Any]] = {}

# ═══════════════════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN EVENTS
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize KALKI on server startup"""
    global copilot
    try:
        logger.info("🚀 Initializing KALKI Construction Copilot...")
        logger.info("⏳ Loading models (this may take 30-60 seconds)...")
        
        copilot = EnhancedConstructionCopilot()
        
        logger.info("✅ KALKI Construction Copilot loaded successfully!")
        logger.info("🧠 All 10 intelligence enhancements active")
        logger.info("🌐 API server ready to accept requests")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize KALKI: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("🛑 Shutting down KALKI API server...")

# ═══════════════════════════════════════════════════════════════════════
# AUTHENTICATION MODELS & UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return user data"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Check if user exists
        user = users_db.get(username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# ═══════════════════════════════════════════════════════════════════════
# AUTHENTICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", response_model=Token)
async def register(user: UserRegister):
    """Register new user"""
    
    # Check if user already exists
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    if any(u.get("email") == user.email for u in users_db.values()):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Truncate password to 72 bytes for bcrypt compatibility (UTF-8 safe)
    pw_bytes = user.password.encode('utf-8')
    if len(pw_bytes) > 72:
        # Only include complete UTF-8 characters up to 72 bytes
        safe_password = pw_bytes[:72]
        while True:
            try:
                safe_password_str = safe_password.decode('utf-8')
                break
            except UnicodeDecodeError:
                safe_password = safe_password[:-1]
    else:
        safe_password_str = user.password
    hashed_password = pwd_context.hash(safe_password_str)
    
    # Create user
    user_data = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "projects": []
    }
    
    users_db[user.username] = user_data
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    # Remove sensitive data
    safe_user = {k: v for k, v in user_data.items() if k != "hashed_password"}
    
    logger.info(f"✅ New user registered: {user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": safe_user
    }

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user"""
    
    # Check if user exists
    user = users_db.get(credentials.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Verify password
    if not pwd_context.verify(credentials.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create access token
    access_token = create_access_token(data={"sub": credentials.username})
    
    # Remove sensitive data
    safe_user = {k: v for k, v in user.items() if k != "hashed_password"}
    
    logger.info(f"✅ User logged in: {credentials.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": safe_user
    }

# ═══════════════════════════════════════════════════════════════════════
# API REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=5000)
    project_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    reasoning: Optional[str] = None
    next_steps: Optional[List[str]] = None
    enhancements_used: List[str]
    timestamp: str

class RoadmapRequest(BaseModel):
    project_type: str = Field(..., pattern="^(adu|remodel|new_construction)$")
    property_data: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None

class RoadmapResponse(BaseModel):
    steps: List[Dict[str, Any]]
    total_weeks: int
    total_cost: float
    timeline: Dict[str, Any]
    complexity_score: float

class ConsensusRequest(BaseModel):
    decision: str = Field(..., min_length=1, max_length=2000)
    context: Dict[str, Any]
    require_unanimous: bool = False

class ConsensusResponse(BaseModel):
    agreement: float
    recommendation: str
    individual_analyses: List[Dict[str, Any]]
    conflicts: List[str]
    voting_breakdown: Dict[str, str]

class PropertyRequest(BaseModel):
    address: str = Field(..., min_length=1, max_length=500)
    property_type: str = Field(..., pattern="^(residential|commercial)$")
    project_intent: str

class PropertyResponse(BaseModel):
    complexity_score: float
    zoning_info: Dict[str, Any]
    setbacks: Optional[Dict[str, Any]] = None
    permit_requirements: List[str]
    estimated_timeline: str
    risks: List[str]

# ═══════════════════════════════════════════════════════════════════════
# MAIN API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest
):
    """
    Main chat endpoint - interact with Construction Copilot
    
    This endpoint processes user queries and returns AI-powered guidance
    using all 10 intelligence enhancements.
    """
    if not copilot:
        raise HTTPException(status_code=503, detail="KALKI not initialized")
    logger.info(f"💬 Chat request: {request.user_input[:50]}...")
    result = await copilot.process_user_input(
        user_input=request.user_input,
        project_id=request.project_id,
        context=request.context or {}
    )
    enhancements_used = []
    if result.get('consciousness_reasoning'):
        enhancements_used.append('consciousness')
    if result.get('consensus_validation'):
        enhancements_used.append('multi_agent_consensus')
    if result.get('meta_learning_applied'):
        enhancements_used.append('meta_learning')
    if result.get('research_conducted'):
        enhancements_used.append('autonomous_research')
    response = ChatResponse(
        response=result.get('response', 'I can help you with your construction project!'),
        confidence=result.get('confidence', 0.8),
        reasoning=result.get('reasoning'),
        next_steps=result.get('next_steps'),
        enhancements_used=enhancements_used,
        timestamp=datetime.utcnow().isoformat()
    )
    logger.info(f"✅ Chat response generated (confidence: {response.confidence})")
    return response

@app.post("/api/roadmap", response_model=RoadmapResponse)
async def generate_roadmap(
    request: RoadmapRequest
):
    """
    Generate detailed construction roadmap
    
    Returns step-by-step guidance with timelines, costs, and milestones.
    """
    if not copilot:
        raise HTTPException(status_code=503, detail="KALKI not initialized")
    logger.info(f"🗺️ Roadmap request: {request.project_type}")
    roadmap = await copilot.roadmap_generator.generate_roadmap(
        project_type=request.project_type,
        property_data=request.property_data,
        user_preferences=request.user_preferences or {}
    )
    response = RoadmapResponse(
        steps=roadmap.get('steps', []),
        total_weeks=roadmap.get('total_weeks', 0),
        total_cost=roadmap.get('total_cost', 0.0),
        timeline=roadmap.get('timeline', {}),
        complexity_score=roadmap.get('complexity_score', 0.5)
    )
    logger.info(f"✅ Roadmap generated: {len(response.steps)} steps, {response.total_weeks} weeks")
    return response

@app.post("/api/consensus", response_model=ConsensusResponse)
async def multi_agent_consensus(
    request: ConsensusRequest
):
    """
    Multi-agent consensus validation
    
    Gets 3 specialized agents (feasibility, quality, innovation) to vote
    on critical construction decisions.
    """
    if not copilot:
        raise HTTPException(status_code=503, detail="KALKI not initialized")
    logger.info(f"🗳️ Consensus request: {request.decision[:50]}...")
    result = await copilot.multi_agent_consensus.analyze(
        decision=request.decision,
        context=request.context,
        require_unanimous=request.require_unanimous,
        agents=['feasibility', 'quality', 'innovation'],
        domain='construction'
    )
    response = ConsensusResponse(
        agreement=result.get('agreement', 0.0),
        recommendation=result.get('recommendation', ''),
        individual_analyses=result.get('individual_analyses', []),
        conflicts=result.get('conflicts', []),
        voting_breakdown={
            'feasibility': result.get('individual_analyses', [{}])[0].get('vote', 'unknown') if len(result.get('individual_analyses', [])) > 0 else 'unknown',
            'quality': result.get('individual_analyses', [{}])[1].get('vote', 'unknown') if len(result.get('individual_analyses', [])) > 1 else 'unknown',
            'innovation': result.get('individual_analyses', [{}])[2].get('vote', 'unknown') if len(result.get('individual_analyses', [])) > 2 else 'unknown',
        }
    )
    logger.info(f"✅ Consensus reached: {response.agreement*100}% agreement")
    return response

@app.post("/api/property", response_model=PropertyResponse)
async def analyze_property(
    request: PropertyRequest
):
    """
    Property intelligence gathering
    
    Analyzes property for zoning, setbacks, permits, and complexity.
    """
    if not copilot:
        raise HTTPException(status_code=503, detail="KALKI not initialized")
    logger.info(f"🏠 Property analysis: {request.address}")
    result = await copilot.property_intelligence.gather_intelligence(
        address=request.address,
        property_type=request.property_type,
        project_intent=request.project_intent
    )
    response = PropertyResponse(
        complexity_score=result.get('complexity_score', 0.0),
        zoning_info=result.get('zoning_info', {}),
        setbacks=result.get('setbacks'),
        permit_requirements=result.get('permit_requirements', []),
        estimated_timeline=result.get('estimated_timeline', 'Unknown'),
        risks=result.get('risks', [])
    )
    logger.info(f"✅ Property analyzed: complexity {response.complexity_score}")
    return response

# ═══════════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "kalki_loaded": copilot is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "models": {
            "text": "llama-3.1-8b-instruct",
            "vision": "llama-3.2-11b-vision"
        },
        "enhancements": [
            "consciousness",
            "meta_learning",
            "autonomous_research",
            "multi_agent_consensus",
            "visual_knowledge_graph",
            "reinforcement_learning",
            "self_evolution",
            "domain_registry",
            "journey_management",
            "property_intelligence"
        ]
    }

@app.get("/api/user/projects")
async def get_user_projects():
    """Get user's construction projects"""
    return {
        "projects": [],
        "count": 0
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An error occurred"
        }
    )

# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Starting KALKI Construction Copilot API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/health")
    print("\n⏳ Loading KALKI models (this may take 30-60 seconds)...\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
