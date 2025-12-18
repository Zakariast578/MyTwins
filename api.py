import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from my_info_agent import ask_question_api
import uvicorn

# -------------------------
# Load .env
# -------------------------
load_dotenv()
frontend_urls = os.getenv("FRONTEND_URL", "")
# Split multiple URLs by comma and strip spaces
origins = [url.strip() for url in frontend_urls.split(",") if url.strip()]

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="My Info Agent API",
    description="Personal AI Agent powered by Gemini + FAISS",
    version="1.0.0"
)

# -------------------------
# Enable CORS
# -------------------------
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
    try:
        answer = ask_question_api(request.question)
        if not answer:
            raise HTTPException(status_code=404, detail="No answer returned from agent")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

