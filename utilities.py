import os
import pdfplumber
from pathlib import Path
import weave

def process_pdf(file_path):
    """
    Opens a PDF file and extracts text from each page.

    Parameters:
    file_path (str): Path to the PDF file.

    Returns:
    tuple: A tuple containing the full extracted text and an error message (if any).
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text is None or page_text == "":
                    return None, "Document could not be read"
                full_text += page_text
            return full_text, None
    except Exception as e:
        return None, str(e)

def get_prompt(prompt_name: str) -> weave.trace.refs.ObjectRef:

    """gets text content from prompt file with given name"""
    prompt_path = Path("prompts") / f"{prompt_name}.txt"
    prompt_text = prompt_path.read_text()

    stringprompt_object = weave.StringPrompt(prompt_text)

    prompt = weave.publish(stringprompt_object, name=prompt_name)

    return prompt