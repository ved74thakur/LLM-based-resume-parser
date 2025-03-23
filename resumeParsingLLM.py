from pdfminer.high_level import extract_text
import json
import re
import os
import openai
from dotenv import load_dotenv
import sys

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(pdf_path: str) -> str:
    text = extract_text(pdf_path)
    return text



#preprocessing
def preprocess_resume_text(text: str) -> str:
    import re

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
            messages = [
                {"role": "system", "content":"You are an assistant that extracts resume features."},
                {"role": "user", "content": prompt}

            ],
            temperature = 0.2
        )
        content = response['choices'][0]['message']['content']

        if content.startswith("``"):
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


def main(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        raw_text = extract_text_from_pdf(file_path)
    else: 
        print("Re-upload")

    
    processed_text = preprocess_resume_text(raw_text)
    features = extract_resume_features(processed_text)
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resumeParsingLLM.py <input_file>")
        sys.exit(1)
    print("Enter file path:")
    file_path = sys.argv[1]
    
    main(file_path)


