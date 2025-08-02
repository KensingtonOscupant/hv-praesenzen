import os
import pdfplumber
from pathlib import Path
import weave

@weave.op()
def preprocess_example(example):
    """
    Opens a PDF file and extracts text from each page.

    Parameters:
    example: Either a string file path or a dictionary with a 'file_path' key.

    Returns:
    dict: A dictionary containing the extracted pages as a list.
    """
    try:
        file_path = example['file_path']
            
        with pdfplumber.open(file_path) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text is not None and page_text != "":
                    pages.append(page_text)
            
            result = {"pages": pages, "error": None}
            return result
    except Exception as e:
        result = {"pages": [], "error": str(e)}
        return result

def get_prompt(prompt_name: str) -> weave.trace.refs.ObjectRef:

    """gets text content from prompt file with given name"""
    prompt_path = Path("prompts") / f"{prompt_name}.txt"
    prompt_text = prompt_path.read_text()

    stringprompt_object = weave.StringPrompt(prompt_text)

    prompt = weave.publish(stringprompt_object, name=prompt_name)

    return prompt