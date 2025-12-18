from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from my_info_agent import ask_question_api

app = FastAPI(
    title="My Info Agent API",
    description="Personal AI Agent powered by Gemini + FAISS",
    version="1.0.0"
)

# -------------------------
# Enable CORS
# -------------------------
origins = [
    "http://localhost:5173",  # React dev server
    "https://zakariasaid.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class QuestionRequest(BaseModel):
    question: str

# -------------------------
# Response schema
# -------------------------
class AnswerResponse(BaseModel):
    answer: str

# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {"status": "My Info Agent API is running"}

# -------------------------
# Ask agent endpoint
# -------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_agent_api(request: QuestionRequest):
    answer = ask_question_api(request.question)
    return {"answer": answer}
