import os
import uuid
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import google.generativeai as genai
import httpx
import asyncio
from mangum import Mangum
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()

try:
    from middleware import ChatSessionMiddleware
except ImportError:
    ChatSessionMiddleware = None

# API keys directly hardcoded (NOT RECOMMENDED FOR PRODUCTION)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        safety_settings = [
             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
         ]

        SYSTEM_INSTRUCTION =  """You are a specialized travel assistant chatbot focused **exclusively** on travel and tourism.

        When responding, include a key named 'image_keyword' in your answer JSON, containing a relevant place or object name for image search.
Example:
{
    "response": "Hunza is a beautiful valley...",
    "image_keyword": "Hunza Valley"
}

    **Core Directives:**
    1.  **Topic Restriction:** You MUST discuss **only** travel-related topics (destinations, activities, booking, planning, culture related to travel, food related to travel, etc.). Politely refuse ANY non-travel questions (e.g., math, general knowledge, news, coding, health advice). Example refusal: "My expertise is strictly in travel and tourism. How can I help you plan a trip today?"

    2.  **Pakistan Travel Focus:** If a user asks for travel plans, recommendations, or itineraries specifically *within Pakistan*, you MUST **exclusively** recommend and detail plans for the **Northern Areas of Pakistan**, including:
        - Hunza, Skardu, Gilgit, Fairy Meadows, Deosai, Swat, Kalam, Malam Jabba, Naran, Kaghan, Chitral, Murree, Galiyat
        - **Islamabad** (travel-related spots only such as Pir Sohawa, Daman-e-Koh, Margalla Hills, Rawal Lake, Faisal Mosque)
        
        Do NOT suggest other regions of Pakistan (like Karachi, Lahore, Bahawalpur, interior Sindh, Southern Punjab) unless the user *explicitly* asks about a non-Northern Area destination *after* you have already enthusiastically focused on the Northern Areas. 

    3.  **General Travel:** For travel requests *outside* Pakistan, or general travel concepts, answer normally within the travel domain, maintaining travel-specific relevance.

    4.  **Image Context:** When describing places, landmarks, or visual activities, provide descriptive details that help users imagine or find a good image (e.g., scenic views, unique architecture, cultural visuals, vibrant colors).
    5.  **Image Context:**  Always map ambiguous or commonly misinterpreted location names (e.g., "Swat") to their full, descriptive forms (e.g., "Swat Valley Pakistan") before using them in image queries to ensure search results reflect the intended geographical or cultural context.

    6.  **Conversational Flow:** Use chat history to maintain continuity and provide relevant follow-ups. Be enthusiastic, friendly, and focused on enhancing the user's travel curiosity and experience.
    7. **Budget Format Restriction:** When discussing or estimating budgets, you must always reply using Pakistani Rupees (PKR) only, regardless of the destination or travel location. Avoid using any other currency in budget estimates
    8.  **Safety:** Where appropriate, include relevant safety advice, travel seasons, weather alerts, road conditions, packing tips, or health precautions related to the destination.

"""

        gemini_model = genai.GenerativeModel(
             model_name="gemini-1.5-flash-latest",
             safety_settings=safety_settings,
             system_instruction=SYSTEM_INSTRUCTION
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize Google Generative AI client: {e}")
        gemini_model = None

app = FastAPI(
    title="Pakistan Northern Areas Travel Chatbot",
    description="A chatbot focused on travel, specifically promoting Northern Pakistan, with history and image features.",
    version="1.0.0",
)

if ChatSessionMiddleware:
    app.add_middleware(ChatSessionMiddleware)

chat_histories: Dict[str, List[Dict[str, Any]]] = {}

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the chatbot.")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Identifier for the current chat session.")
    bot_response: str = Field(..., description="The chatbot's textual response.")
    image_urls: List[str] = Field(default=[], description="List of relevant image URLs (up to 3).")

def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]

def add_message_to_history(session_id: str, role: str, text: str):
    if role not in ["user", "model"]:
        role = "user"

    history = get_chat_history(session_id)
    history.append({"role": role, "parts": [text]})

    MAX_HISTORY_TURNS = 10
    MAX_HISTORY_LENGTH = MAX_HISTORY_TURNS * 2
    if len(history) > MAX_HISTORY_LENGTH:
        chat_histories[session_id] = history[-MAX_HISTORY_LENGTH:]

async def get_images_from_unsplash(query: str, count: int = 3) -> List[str]:
    if not UNSPLASH_ACCESS_KEY:
        return []
    if not query:
        return []
    if count <= 0:
        return []

    UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1"
    }
    params = {
        "query": query,
        "per_page": count,
        "orientation": "landscape"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(UNSPLASH_API_URL, headers=headers, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            image_urls = [
                result["urls"]["regular"]
                for result in results
                if isinstance(result, dict) and "urls" in result and "regular" in result["urls"]
            ]
            return image_urls[:count]
    except Exception as e:
        print(f"ERROR: Unsplash fetch error: {e}") # Keep minimal error prints?
        return []

async def get_chatbot_response_gemini(session_id: str, user_message: str) -> Tuple[str, str]:
    if not gemini_model:
        return "I'm currently unable to process requests. Please try again shortly.", ""

    history = get_chat_history(session_id)
    messages_for_api = history + [{"role": "user", "parts": [user_message]}]

    try:
        response = await gemini_model.generate_content_async(messages_for_api)

        if response.parts:
            bot_response_text = response.text
        else:
            block_reason = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name
            if block_reason == 'SAFETY':
                 return "I cannot provide a response that goes against safety guidelines. Could we talk about a different travel topic?", ""
            else:
                 return "I seem to have trouble generating a response for that. Could you rephrase or ask about something else in travel?", ""

        image_search_query = ""
        northern_areas = ["Hunza", "Murree","Islamabad", "Skardu", "Gilgit", "Naran", "Kaghan", "Swat", "Chitral", "Fairy Meadows", "Deosai", "Passu Cones", "Attabad Lake", "Rakaposhi", "Eagle's Nest"]
        general_keywords = ["mountains", "lake", "valley", "trekking", "glacier", "peak", "landscape", "bazar", "fort"]

        response_lower = bot_response_text.lower()
        found_northern = [area for area in northern_areas if area.lower() in response_lower]
        found_general = [kw for kw in general_keywords if kw in response_lower]

        if found_northern:
            location = found_northern[0]
            if location.lower() == "swat":
                image_search_query = "Swat Valley Pakistan"
            elif location.lower() == "hunza":
                image_search_query = "Hunza Valley Pakistan"
            elif location.lower() == "gilgit":
                image_search_query = "Gilgit Baltistan Pakistan"
            elif location.lower() == "fairy meadows":
                image_search_query = "Fairy Meadows Nanga Parbat Pakistan"
            # elif location.lower() == "Islamabad":
            #     image_search_query = "Faisal Mosque Daman-e-Koh Pakistan Monument lake view park"
            else:
                image_search_query = location
                if "valley" not in image_search_query.lower():
                    image_search_query += " Valley"
                image_search_query += " Pakistan"
        elif found_general:
             image_search_query = found_general[0] + " travel"

        return bot_response_text, image_search_query

    except Exception as e:
        print(f"ERROR: Gemini API call error: {e}") # Keep minimal error prints?
        return "An unexpected error occurred while processing your request. Please try again.", ""

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request_body: ChatRequest,
    request: Request
):
    try:
        session_id = request.state.session_id
    except AttributeError:
        raise HTTPException(status_code=500, detail="Session configuration error.")

    user_message = request_body.message
    get_chat_history(session_id)
    add_message_to_history(session_id, "user", user_message)
    bot_response_text, image_search_query = await get_chatbot_response_gemini(session_id, user_message)

    if not bot_response_text.startswith("I'm currently unable") and not bot_response_text.startswith("An unexpected error"):
         add_message_to_history(session_id, "model", bot_response_text)

    image_urls = await get_images_from_unsplash(image_search_query, count=3)

    response_data = ChatResponse(
    session_id=session_id,
    bot_response=bot_response_text,
    image_urls=image_urls
)

    return response_data

@app.get("/")
async def root():
    return {"message": "Welcome to the Pakistan Northern Areas Travel Chatbot API!"}

@app.get("/sessions")
def get_sessions():
    return {"active_sessions": list(chat_histories.keys())}


@app.get("/health", status_code=200)
async def health_check():
    ai_status = "error"
    if gemini_model:
        ai_status = "initialized"
    elif not GOOGLE_API_KEY:
        ai_status = "disabled_no_key"
    return {"status": "ok", "ai_status": ai_status}

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "Hello from FastAPI on Vercel!"})

if __name__ == "__main__":
    import uvicorn
    print("Starting Travel Chatbot API server...") # Keep startup message?
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


handler = Mangum(app)