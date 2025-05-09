from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chatbot_response(msg: Message):
    reply = f"Bot: You said '{msg.message}'"
    return JSONResponse(content={"response": reply})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
