from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from .agent import agent  # относительный импорт из agent.py
import os

app = FastAPI(title="Fitness Trainer Agent")

# ✅ CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API эндпоинт для инференса ===
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/generate", response_model=ChatResponse)
async def generate(req: ChatRequest):
    try:
        result = agent.invoke({"messages": [HumanMessage(content=req.message)]})
        return ChatResponse(response=result["messages"][-1].content)
    except Exception as e:
        return ChatResponse(response=f"Ошибка: {str(e)}")

# === Статика: отдаём HTML-интерфейс ===
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Отдаёт веб-интерфейс при открытии корня сайта"""
    ui_path = os.path.join(os.path.dirname(__file__), "inference.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse(content="<h1>⚠️ inference.html не найден</h1><p>Поместите файл в папку lab3/</p>")
