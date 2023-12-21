import os
import pdfplumber
import json
import wandb

# Load the config file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

config = load_config('config.json')

def calculate_cost(input_tokens, output_tokens):
    input_price_per_1000 = config['token_cost']['input_price_per_1000']
    output_price_per_1000 = config['token_cost']['output_price_per_1000']

    # Calculate the cost for input tokens
    input_cost = (input_tokens / 1000) * input_price_per_1000

    # Calculate the cost for output tokens
    output_cost = (output_tokens / 1000) * output_price_per_1000

    # Total cost
    return input_cost + output_cost

def find_subdirectory(directory, id_value):
    """
    Finds and returns the subdirectory path within the given directory that ends with the specified id_value.
    """
    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        if os.path.isdir(subdirectory_path) and subdirectory.endswith(id_value):
            return os.path.join(subdirectory_path, "ASM")
    return None

def find_matching_files(directory_path, file_suffix):
    """
    Searches for all files with a specific suffix in the given directory.

    Parameters:
    directory_path (str): The path to the directory where the search is performed.
    file_suffix (str): The suffix of the files to find (e.g., last two digits of the year for PDF files).

    Returns:
    list[str]: A list of paths to the found files. Empty if no files are found.
    """
    matching_files = []
    if directory_path:
        for file in os.listdir(directory_path):
            if file.endswith(file_suffix + ".pdf"):
                matching_files.append(os.path.join(directory_path, file))
    return matching_files

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

def call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost):
    """
    Makes a call to the GPT language model and returns the response and the cost.

    Parameters:
    client (obj): The GPT client object.
    system_prompt (str): The system prompt for the GPT model.
    user_prompt (str): The user prompt for the GPT model.
    full_text (str): The text to be analyzed by the GPT model.
    calculate_cost (function): Function to calculate the cost of the GPT call.

    Returns:
    tuple: A tuple containing the response from GPT and the cost of the call.
    """
    combined_prompt = user_prompt + full_text
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_prompt}
        ],
        temperature=0.2
    )

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(input_tokens, output_tokens)

    return response.choices[0].message.content, cost

