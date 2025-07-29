import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


class Prompt(BaseModel):
    text: str


@app.post("/ask")
def ask_gemini(prompt: Prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt.text
    )
    return {"reply": response.text}
