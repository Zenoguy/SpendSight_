from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import create_vector_store, query_rag

app = FastAPI()

class CreateSessionRequest(BaseModel):
    user_id: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    user_id: str

@app.post("/create-session")
def create_session(request: CreateSessionRequest):
    """Create RAG session for user"""
    session_id = create_vector_store(request.user_id)
    
    if not session_id:
        raise HTTPException(status_code=404, detail="No OCR documents found for user")
    
    return {"session_id": session_id, "message": "Session created successfully"}

@app.post("/query")
def query(request: QueryRequest):
    """Query RAG system"""
    result = query_rag(request.session_id, request.query, request.user_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result
