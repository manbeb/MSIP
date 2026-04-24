from fastapi import FastAPI
from pydantic import BaseModel
from agent import agent_executor

app = FastAPI(title="Fitness Trainer Agent")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/generate", response_model=ChatResponse)
async def generate(req: ChatRequest):
    try:
        result = agent_executor.invoke({"input": req.message})
        return ChatResponse(response=result["output"])
    except Exception as e:
        return ChatResponse(response=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
