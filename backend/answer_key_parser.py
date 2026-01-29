import pdfplumber
import re
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer



def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text



def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()



def extract_keywords_tfidf(text, top_n=5):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1)
    )

    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    word_scores = list(zip(feature_names, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, score in word_scores[:top_n]]



def parse_answer_key(text):
    pattern = r'(Q\d+\..*?)(?=Q\d+\.|$)'
    blocks = re.findall(pattern, text, re.DOTALL)

    answer_key = {}

    for block in blocks:
        q_no_match = re.search(r'(Q\d+)\.', block)
        if not q_no_match:
            continue

        q_no = q_no_match.group(1)

        marks_match = re.search(r'Max Marks\s*:\s*(\d+)', block)
        if not marks_match:
            continue

        max_marks = int(marks_match.group(1))

        answer_text = re.sub(r'Q\d+\.', '', block)
        answer_text = re.sub(r'\(.*?Max Marks\s*:\s*\d+.*?\)', '', answer_text)
        answer_text = answer_text.strip()

        keywords = extract_keywords_tfidf(answer_text, top_n=5)

        answer_key[q_no] = {
            "max_marks": max_marks,
            "model_answer": answer_text,
            "keywords": keywords
        }

    return answer_key



def parse_answer_key_pdf(pdf_path):
    """
    Used by FastAPI backend.
    Takes PDF path → returns answer key JSON (dict)
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    answer_key = parse_answer_key(cleaned_text)
    return answer_key



def create_answer_key_json(pdf_path, output_path):
    answer_key = parse_answer_key_pdf(pdf_path)

    if not answer_key:
        print("❌ No questions parsed. Check PDF format.")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=4, ensure_ascii=False)

    print("✅ Answer key JSON created successfully!")
    print(f"📁 Output file: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python answer_key_parser.py <answer_key.pdf> <output.json>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2]

    create_answer_key_json(pdf_path, output_path)
