from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Ensure API keys are available
if not GEMINI_API_KEY or not DEEPGRAM_API_KEY:
    raise ValueError("Missing API keys! Check your .env file.")

app = FastAPI()

# Allow frontend to access API (CORS policy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    prompt: str

class NarrationRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_story(request: StoryRequest):
    """Generate a story using the Gemini API."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": request.prompt}]}]},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API Error: {response.text}")

        response_data = response.json()

        # Extract the story text
        story = (
            response_data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No story generated.")
        )
        return {"story": story}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Gemini API: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    
@app.post("/narrate")
async def narrate_story(request: NarrationRequest):
    """Convert generated text to speech using Deepgram API."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required for narration")

    try:
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000"

        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {"text": request.text}

        response = requests.post(url, headers=headers, json=data)

        print("Deepgram API Response Code:", response.status_code)  # Debugging
        print("Deepgram API Response Headers:", response.headers)  # Debugging

        if response.status_code != 200:
            return {"error": "Deepgram API Error", "details": response.text}

        # Save and return the audio file
        audio_path = "output.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)

        return FileResponse(audio_path, media_type="audio/wav", filename="narration.wav")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Deepgram API: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
