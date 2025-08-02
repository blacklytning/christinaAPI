import os
import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
from supabase import Client, create_client

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
supabase_url = os.getenv("SUPABASE_URL")
supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

# Initialize clients
client = genai.Client(api_key=api_key)
supabase: Client = create_client(supabase_url, supabase_anon_key)

media_dir = Path("media")
media_dir.mkdir(exist_ok=True)

# Extensions considered code (to be renamed to .txt)
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".html",
    ".css",
    ".json",
    ".xml",
    ".sh",
    ".rb",
    ".php",
    ".go",
    ".rs",
    ".sql",
    ".yaml",
    ".yml",
}


def rename_if_code(file_path: Path) -> Path:
    ext = file_path.suffix.lower()
    if ext in CODE_EXTENSIONS:
        txt_path = file_path.with_suffix(".txt")
        file_path.rename(txt_path)
        return txt_path
    return file_path


class Prompt(BaseModel):
    text: str


class User(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None


# Authentication dependency
async def get_current_user(authorization: str = Header(None)) -> User:
    if not authorization:
        raise HTTPException(
            status_code=401, detail="Authorization header missing")

    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")

        # Get user from Supabase using the token
        response = supabase.auth.get_user(token)
        user_data = response.user

        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid token")

        return User(
            id=user_data.id,
            email=user_data.email,
            name=user_data.user_metadata.get("full_name"),
            avatar_url=user_data.user_metadata.get("avatar_url"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {str(e)}")


# Optional authentication (for endpoints that can work with or without auth)
async def get_current_user_optional(
    authorization: str = Header(None),
) -> Optional[User]:
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


@app.get("/auth/user")
async def get_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {"user": current_user}


@app.post("/ask")
def ask_gemini(prompt: Prompt, current_user: User = Depends(get_current_user)):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt.text
    )
    return {"reply": response.text}


@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    uploaded_files = []

    for file in files:
        original_path = media_dir / file.filename
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            final_path = rename_if_code(original_path)
            uploaded_file = client.files.upload(file=final_path)
            uploaded_files.append(uploaded_file)
        except Exception as e:
            return {"error": f"Error uploading file {file.filename}: {str(e)}"}

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[*uploaded_files, "\n\n", prompt],
        )
        return {"reply": response.text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze-screen")
async def analyze_screen(
    screenshot: UploadFile = File(...),
    prompt: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    try:
        print(f"Received prompt: {prompt}")
        print(
            f"Screenshot file: {screenshot.filename}, content type: {
                screenshot.content_type}"
        )

        # Save screenshot temporarily
        screenshot_path = media_dir / f"screenshot_{screenshot.filename}"
        with open(screenshot_path, "wb") as f:
            shutil.copyfileobj(screenshot.file, f)

        print(f"Screenshot saved to: {screenshot_path}")

        # Upload to Gemini
        uploaded_file = client.files.upload(file=screenshot_path)
        print(f"File uploaded to Gemini: {uploaded_file}")

        # Generate response with image analysis capabilities
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                uploaded_file,
                f"Analyze this screenshot and help with: {
                    prompt}. Focus on any code, errors, UI elements, or technical content visible. Provide specific and actionable advice based on what you can see.",
            ],
        )

        # Clean up
        screenshot_path.unlink()

        return {"reply": response.text}
    except Exception as e:
        print(f"Error in analyze_screen: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
