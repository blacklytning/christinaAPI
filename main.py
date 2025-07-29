import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
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


@app.post("/ask")
def ask_gemini(prompt: Prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt.text
    )
    return {"reply": response.text}


media_dir = Path("media")
media_dir.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), prompt: str = Form(...)):
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
