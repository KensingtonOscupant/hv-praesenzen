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
            
            full_text = ""
            failed_pages = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text is not None and page_text.strip() != "":
                        full_text += page_text
                    # Note: We don't treat empty pages as errors, just skip them
                except Exception as page_error:
                    failed_pages.append(f"Page {page_num}: {str(page_error)}")
                    continue
            
            # If we couldn't extract text from any pages
            if full_text == "" and len(failed_pages) > 0:
                return "", f"Failed to extract text from all pages in {actual_path}. Errors: {'; '.join(failed_pages)}"
            elif full_text == "":
                return "", f"No text content found in PDF: {actual_path}"
            
            return full_text, None
            
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

@weave.op()
def get_metadata(file_path: str):
    """
    Extract metadata from file path structure.
    
    Args:
        file_path: Path like "data/Praesenzen_hv-info/Company Name-ID/ASM/HV-Beschluss..."
        
    Returns:
        dict with keys: company_name, key_identity_id, date, ordentlich
    """
    from pathlib import Path
    import re
    
    path_obj = Path(file_path)
    path_parts = path_obj.parts
    
    # Extract company info from third level (index 2)
    if len(path_parts) < 3:
        raise ValueError(f"Path does not have enough levels: {file_path}")
    
    company_level = path_parts[2]  # Third level: "Fresenius Medical Care AG & Co. KGaA-14830"
    
    # Find the last dash to split company name from ID
    last_dash_index = company_level.rfind('-')
    if last_dash_index == -1:
        raise ValueError(f"No dash found in company level: {company_level}")
    
    company_name = company_level[:last_dash_index]
    key_identity_id = company_level[last_dash_index + 1:]
    
    # Extract date and ordentlich from PDF filename
    pdf_filename = path_obj.name  # "HV-Beschluss zur ordentlichen Hauptversammlung am 11.05.10.pdf"
    
    # Extract date using regex pattern DD.MM.YY
    date_pattern = r'\b(\d{2}\.\d{2}\.\d{2})\b'
    date_match = re.search(date_pattern, pdf_filename)
    if not date_match:
        raise ValueError(f"No date pattern DD.MM.YY found in filename: {pdf_filename}")
    
    date = date_match.group(1)
    
    # Determine if it's ordentlich or außerordentlich
    # Check for außerordentlichen first since it contains ordentlichen as substring
    if "außerordentlichen" in pdf_filename:
        ordentlich = False
    elif "ordentlichen" in pdf_filename:
        ordentlich = True
    else:
        raise ValueError(f"Neither 'ordentlichen' nor 'außerordentlichen' found in filename: {pdf_filename}")
    
    return {
        "company_name": company_name,
        "key_identity_id": key_identity_id,
        "date": date,
        "ordentlich": ordentlich
    }