# Handwritten Answer Evaluation System

An automated system for evaluating handwritten exam answers using OCR and LLM-based grading.

## Features

- **OCR Integration**: Extract text from handwritten answer papers using OCR API
- **PDF Processing**: Parse answer key and student answers from PDF documents
- **LLM-Based Evaluation**: Uses Groq API with GPT model for intelligent grading
- **Automated Marking**: Assigns marks based on conceptual correctness
- **PDF Report Generation**: Generate evaluation reports as PDF

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── evaluator.py         # LLM-based evaluation logic
│   └── answer_key_parser.py # Answer key PDF parsing
├── frontend/
│   └── index.html           # Web UI
└── answer_key.pdf           # Sample answer key
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install fastapi uvicorn groq pdfplumber reportlab python-multipart
   ```

## Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
OCR_API_URL=your_ocr_service_url
```
## Running the OCR
1.copy the code in colab notebook

2. set your ngrok auth token

3. run the code in google colab

4. replace OCR_API_URL with your url

## Running the Application

1. Start the FastAPI server:
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```

2. Open the frontend:
   - Navigate to `http://localhost:8000/docs` for API documentation
   - Or open `frontend/index.html` in a browser

## API Endpoints

- **POST /evaluate**: Submit answer key PDF and answer paper PDF for evaluation

## How It Works

1. User uploads answer key PDF and student answer paper PDF
2. Answer key is parsed to extract questions and expected answers
3. Student answer paper is sent to OCR service for text extraction
4. LLM evaluates student answers against the answer key
5. Marks are assigned with feedback and missing points
6. Results are returned as JSON and PDF report
