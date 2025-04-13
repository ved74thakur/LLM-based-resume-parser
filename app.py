from flask import Flask, request, jsonify
import os
import tempfile
import json
import re
import openai
from pdfminer.high_level import extract_text
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = extract_text(pdf_path)
    return text

def preprocess_resume_text(text: str) -> str:
    # Remove non-ascii characters and compress whitespace.
    text = text.encode('ascii', errors='ignore').decode()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_resume_features(preprocessed_text: str) -> dict:
    prompt = f"""
    Extract the following key features from the preprocessed_text:
    1. Skills: Provide the list of skills mentioned. 
    2. Job Roles: Provide a list of all job titles or roles mentioned.
    3. Total Experience: Sum up the total work experiences (in years) from all jobs.
    
    Return the output as a JSON object with keys:
    "skills", "job_roles", "total_experience", and "jobs".

    Resume Text:
    {preprocessed_text}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts resume features."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response['choices'][0]['message']['content']

        # Remove markdown formatting if it exists.
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)
    except Exception as e:
        data = {
            "error": "Failed to parse JSON output",
            "raw_output": content if 'content' in locals() else "",
            "exception": str(e)
        }
    return data

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    # Save the file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name

        # Extract and preprocess text
        raw_text = extract_text_from_pdf(tmp_file_path)
        processed_text = preprocess_resume_text(raw_text)
        features = extract_resume_features(processed_text)
    finally:
        # Remove the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return jsonify(features)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'working'})

    
if __name__ == '__main__':
    app.run(debug=True)
