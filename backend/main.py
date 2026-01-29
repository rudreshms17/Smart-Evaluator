from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import requests
import shutil
import json
import os
from pathlib import Path
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from evaluator import evaluate_full_text


app = FastAPI(
    title="Handwritten Answer Evaluation API",
    description="OCR + LLM-based Evaluation",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_API_URL = os.getenv("OCR_API_URL")
if not OCR_API_URL:
    raise RuntimeError("OCR_API_URL environment variable not set. Please set it in .env file.")

TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text.strip()


def generate_pdf_report(evaluation_result: dict) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(x, y, "Handwritten Answer Evaluation Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(x, y, "Generated Evaluation Output:")
    y -= 20

    pdf.setFont("Courier", 9)
    pretty_text = json.dumps(evaluation_result, indent=2, ensure_ascii=False)

    for line in pretty_text.split("\n"):
        if y < 40:
            pdf.showPage()
            pdf.setFont("Courier", 9)
            y = height - 40

        pdf.drawString(x, y, line[:100])  
        y -= 12

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return buffer.read()


@app.post("/evaluate")
async def evaluate(
    answer_key: UploadFile = File(...),
    answer_paper: UploadFile = File(...)
):
    try:
        ak_path = TEMP_DIR / "answer_key.pdf"
        ap_path = TEMP_DIR / "answer_paper.pdf"

        with open(ak_path, "wb") as f:
            shutil.copyfileobj(answer_key.file, f)

        with open(ap_path, "wb") as f:
            shutil.copyfileobj(answer_paper.file, f)

        with open(ap_path, "rb") as f:
            response = requests.post(
                OCR_API_URL,
                files={"file": f},
                timeout=600
            )

        if response.status_code != 200:
            raise HTTPException(500, "OCR service failed")

        ocr_json = response.json()

        lines = ocr_json.get("combined_output", [])
        handwritten_text = "\n".join(lines).strip()

        if not handwritten_text:
            raise HTTPException(500, "OCR returned empty text")

        
        (OUTPUT_DIR / "ocr_raw.json").write_text(
            json.dumps(ocr_json, indent=4), encoding="utf-8"
        )
        (OUTPUT_DIR / "handwritten_text.txt").write_text(
            handwritten_text, encoding="utf-8"
        )

        
        answer_key_text = extract_text_from_pdf(ak_path)
        if not answer_key_text:
            raise HTTPException(500, "Answer key extraction failed")

        (OUTPUT_DIR / "answer_key_text.txt").write_text(
            answer_key_text, encoding="utf-8"
        )


        result = evaluate_full_text(
            answer_key_text=answer_key_text,
            student_text=handwritten_text
        )

        (OUTPUT_DIR / "evaluation_result.json").write_text(
            json.dumps(result, indent=4), encoding="utf-8"
        )

        
        pdf_bytes = generate_pdf_report(result)

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=evaluation_report.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
