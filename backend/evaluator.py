

import json
import os
from groq import Groq

# Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set. Please set it in .env file.")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"


def _safe_json_load(raw_text: str) -> dict:
    """
    Attempts to parse JSON.
    If parsing fails, asks LLM to repair JSON once.
    """

    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")

    candidate = raw_text[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:

        repair_prompt = f"""
Fix the following text so it becomes VALID JSON.

Rules:
- Use double quotes only
- Add missing commas
- No trailing commas
- No explanations
- Output ONLY JSON

TEXT:
{candidate}
"""

        repair_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You repair broken JSON. Output ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": repair_prompt
                }
            ],
            temperature=0
        )

        fixed = repair_response.choices[0].message.content.strip()

        start = fixed.find("{")
        end = fixed.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("JSON repair failed")

        return json.loads(fixed[start:end])


def evaluate_full_text(answer_key_text: str, student_text: str) -> dict:
    """
    Evaluates unstructured handwritten answers against unstructured answer key.
    LLM detects questions like 1, 1a, 2b and assigns marks.
    """

    prompt = f"""
You are an automated exam evaluator.

MANDATORY GRADING RULES:
1. Identify ALL questions and sub-questions (1, 1a, 1b, 2a, 2b, etc.)
2. Match student answers ONLY to the correct questions
3. Evaluate the student’s answers using the answer key. If an answer does not directly match, assess whether it is still conceptually and subjectively correct.
4. Be lineant
5. Do NOT hallucinate content
6. NEVER award less than 70% of max marks for a question (very important note)
7. If the answer is weak or incorrect:
   - List only very few missing points (like 2-3 missing points only ignore the rest of the missing points) from the answer key
8. Finally sum up all the marks you have awarded accurately and show the total marks awarded for the answer paper compulsorily

ANSWER KEY:
------------
{answer_key_text}

STUDENT ANSWERS:
----------------
{student_text}

OUTPUT RULES:
- Output ONLY valid JSON
- Use double quotes
- No trailing commas
- No explanations
- No markdown

RETURN JSON in EXACT format:
{{
  "total_marks": number,
  "question_wise_results": {{
    "1a": {{
      "marks_awarded": number,
      "max_marks": number,
      "missing_points": [] 
    }},
    "2b": {{
      "marks_awarded": number,
      "max_marks": number,
      "missing_points": []  
    }}
  }}
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator.\n"
                    "Rules:\n"
                    "- Output ONLY valid JSON\n"
                    "- Use double quotes only\n"
                    "- No trailing commas\n"
                    "- No extra text"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    return _safe_json_load(raw_output)
