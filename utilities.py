import os
import pdfplumber
from pathlib import Path
import weave

@weave.op()
def process_pdf(file_path):
    # Handle case where file_path might be a dictionary
    if isinstance(file_path, dict):
        if 'file_path' not in file_path:
            return [], "Input dictionary missing 'file_path' key"
        actual_path = file_path['file_path']
    else:
        actual_path = file_path
    
    # Validate input type
    if not isinstance(actual_path, (str, Path)):
        return [], f"Invalid file path type: {type(actual_path)}. Expected string or Path object"
    
    # Convert to Path object for easier handling
    path_obj = Path(actual_path)
    
    # Check if file path exists
    if not path_obj.exists():
        return [], f"File does not exist: {actual_path}"
    
    # Check if path is actually a file (not a directory)
    if not path_obj.is_file():
        return [], f"Path is not a file: {actual_path}"
    
    # Check file extension
    if path_obj.suffix.lower() != '.pdf':
        return [], f"File is not a PDF (extension: {path_obj.suffix}): {actual_path}"
    
    # Check if file is empty
    if path_obj.stat().st_size == 0:
        return [], f"PDF file is empty: {actual_path}"
    
    # Check file permissions
    if not os.access(actual_path, os.R_OK):
        return [], f"No read permission for file: {actual_path}"
    
    try:
        with pdfplumber.open(actual_path) as pdf:
            # Check if PDF has any pages
            if len(pdf.pages) == 0:
                return [], f"PDF file contains no pages: {actual_path}"
            
            pages = []
            failed_pages = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text is not None and page_text.strip() != "":
                        pages.append(page_text)
                    # Note: We don't treat empty pages as errors, just skip them
                except Exception as page_error:
                    failed_pages.append(f"Page {page_num}: {str(page_error)}")
                    continue
            
            # If we couldn't extract text from any pages
            if len(pages) == 0 and len(failed_pages) > 0:
                return [], f"Failed to extract text from all pages in {actual_path}. Errors: {'; '.join(failed_pages)}"
            elif len(pages) == 0:
                return [], f"No text content found in PDF: {actual_path}"
            
            return pages, None
            
    except FileNotFoundError:
        return [], f"File not found during processing: {actual_path}"
    except PermissionError:
        return [], f"Permission denied when accessing file: {actual_path}"
    except pdfplumber.exceptions.PDFSyntaxError:
        return [], f"Invalid or corrupted PDF file: {actual_path}"
    except pdfplumber.exceptions.PasswordProtected:
        return [], f"PDF file is password protected: {actual_path}"
    except MemoryError:
        return [], f"Insufficient memory to process large PDF file: {actual_path}"
    except OSError as os_error:
        return [], f"Operating system error when accessing file {actual_path}: {str(os_error)}"
    except Exception as e:
        # Catch any other unexpected errors
        error_type = type(e).__name__
        return [], f"Unexpected error processing PDF {actual_path} ({error_type}): {str(e)}"

def get_prompt(prompt_name: str) -> weave.trace.refs.ObjectRef:

    """gets text content from prompt file with given name"""
    prompt_path = Path("prompts") / f"{prompt_name}.txt"
    prompt_text = prompt_path.read_text()

    stringprompt_object = weave.StringPrompt(prompt_text)

    prompt = weave.publish(stringprompt_object, name=prompt_name)

    return prompt