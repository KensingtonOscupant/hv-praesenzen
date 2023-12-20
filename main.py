import pdfplumber
from tqdm import tqdm
import ast
import re
from collections import Counter
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import wandb
import json
import argparse

load_dotenv()

# configure command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

# set up OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the config file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

config = load_config('config.json')

# Load the CSV file (test data or full data)
df = pd.read_csv(config['csv_file_path'])

# Set the base directory for where the folders with the PDFs are located
directory = config['directory']

# Initialize Weights & Biases
wandb.init(project=config['wandb_project_name'])

# Define W&B Table to store results
columns = config['wandb_table_columns']
table = wandb.Table(columns=columns)

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

def find_file_in_directory(directory_path, file_suffix):
    """
    Searches for a file with a specific suffix in the given directory.

    Parameters:
    directory_path (str): The path to the directory where the search is performed.
    file_suffix (str): The suffix of the file to find (e.g., year value for PDF files).

    Returns:
    str: The path to the found file, or None if no file is found.
    """
    if directory_path:
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) and file.endswith(file_suffix):
                return file_path
    return None

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

# log a row of data to the W&B table
def log_to_wandb(data):
    table.add_data(*data.values())

# Assuming args.start and args.end are defined (the row numbers to start and end with)
start_index = args.start if args.start is not None else 0
end_index = args.end if args.end is not None else len(df)

# Calculate the total number of rows to be processed
total_rows = end_index - start_index

# Iterate over the DataFrame slice with tqdm
for index, row in tqdm(df.iloc[start_index:end_index].iterrows(), total=total_rows):

    """
    Initialize the default data dictionary for each iteration. 
    This will be updated with the results of each step and then logged to W&B.
    """
    data = {
        "ID_Key_original": str(int(row['ID_Key_original'])),
        "Year_original": str(int(row['Year_original'])),
        "Presence_enhanced": row['Presence_enhanced'],
        "Presence_predicted": None,
        "correct": None,
        "error": None,
        "cost": 0,
        "file_path": None,
        "comment": ""
    }

    # Initialize variables

    # error variable
    error = False

    # error during page assessment
    error_during_page_assessment = False

    # alternative document structure, i.e., no table but more like a list
    alternative_document_structure = False

    # total cost pf processing the document
    cost = 0

    # standard deviation of the highest percentage of each document to the other values for the same document
    std_dev = 0

    # mean of the highest percentage of each document to the other values for the same document
    mean = 0

    # add a comment collection variable for the case some pages dont output a number
    comment_collection = ""

    prediction_correct = False

    id_value = str(int(row['ID_Key_original']))
    year_value = str(int(row['Year_original']))

    # find the subdirectory where all reports for the same ID are located
    subdirectory_path = find_subdirectory(directory, id_value)

    # find the file with the year value in the subdirectory
    file_suffix = year_value[-2:] + ".pdf"
    file_path = find_file_in_directory(subdirectory_path, file_suffix)

    if file_path:
        data["file_path"] = file_path
        full_text, error_message = process_pdf(file_path)

        # Check if there was an error processing the PDF
        if error_message:
            data.update({
                "error": error_message,
                "comment": "Error processing PDF document"
            })
            log_to_wandb(data)
            continue

    # In case of file not found
    else:
        data.update({
            "error": "File not found",
            "comment": f"No file found for ID {id_value}"
        })
        log_to_wandb(data)
        continue

    highest_percentage_list = []

    system_prompt = "Du bist ein hilfreicher Assistent, der Berichte von Hauptversammlungen auswertet."
    print(system_prompt)
    user_prompt = "Im folgenden erhältst du einen Bericht einer Hauptversammlung. Das Dokument enthält eine Tabelle mit einer Kopfzeile, aber die Kopfzeile ist beim Extrahieren des Texts beschädigt worden. Bitte gib mir die volle Bezeichnung jeder Spalte in der korrekten Reihenfolge, wie sie im Dokument auftaucht. Antworte ausschließlich mit einer Liste im Format [spalte_1, spalte_2, spalte_3]. Es kann auch eine Kopfzeile mit Multi-Index sein. Wenn du keine Kopfzeile finden kannst, antworte mit [0]. Bericht: "
    print(user_prompt)

    header_row_list = []

    for i in range(1):
        response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
        header_row_list.append(response)
        data['cost'] += cost

    header_counts = Counter(header_row_list)

    most_common_count = header_counts.most_common(1)[0][1]  # Get the count of the most common string

    # Find all strings that have this highest count
    most_common_headers = [header for header, count in header_counts.items() if count == most_common_count]

    # # Handle the situation based on the number of most common strings
    if len(most_common_headers) == 1:
        # Only one most common string
        most_common_header = most_common_headers[0]
        print("Most common header:", most_common_header)
    elif len(most_common_headers) > 1:
        # Multiple strings with the same highest frequency
        print("Multiple headers are equally common:", most_common_headers)
        data.update({
            "error": "Multiple common headers",
            "comment": most_common_headers
        })
        log_to_wandb(data)
        continue
    else:
        data.update({
            "error": "No header row found"
        })
        continue

    # Regular expression pattern to check if string starts with '[' and ends with ']'
    pattern = r'^\[.*\]$'

    # Check if the response string matches the pattern
    if re.match(pattern, most_common_header):
        # Process the string if it matches
        column_names = most_common_header # Adjust as needed
        print("column names: " + column_names)

        # check if response is 0. In that case, we check if the file might not be structured like a table, but more like a list (alternative document structure)
        if column_names == "[0]":

            user_prompt = "Im folgenden erhältst du einen Bericht einer Hauptversammlung. Werden in dem Dokument wiederholt und ausdrücklich Angaben zum auf der Versammlung vertretenen Grundkapital in Prozent gemacht (Also bspw. 'Grundkapital: 30%'? Antworte nur mit [1] oder [0]. Bericht: "

            response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
            header_row_list.append(response)
            data['cost'] += cost

            # check response
            if response == "[1]":

                alternative_document_structure = True

                # find the highest grundkapital percentage in the document
                user_prompt = "Im folgenden erhältst du einen Bericht von einer Hauptversammlung. Antworte ausschließlich mit einer Liste im Format [zahl_1, zahl_2, zahl_3], die ausschließlich alle die genannten Prozentzahlen enthält, die sich auf den Prozentsatz des auf der Hauptversammlung vertretenen Grundkapitals beziehen. Wenn du dir nicht absolut sicher bist, antworte mit [0]. Verwende Punkt statt Komma für die Zahlen. Bericht: "

                response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
                header_row_list.append(response)
                data['cost'] += cost

            elif response == "[0]":
                data.update({
                    "error": "Header row not found",
                })
                log_to_wandb(data)
                continue

            else:
                data.update({
                    "error": "Unexpected response from GPT model to the question about alternative document structure",
                    "comment": response,
                })
                log_to_wandb(data)
                continue

    else:
        data.update({
            "error": "Unexpected response from GPT model to the question about the header row",
            "comment": response,
        })
        log_to_wandb(data)
        continue

    if alternative_document_structure == False:
        user_prompt = "Im folgenden erhältst du einen Bericht einer Hauptversammlung. Das Dokument enthält eine Tabelle mit einer Kopfzeile, aber die Kopfzeile ist beim Extrahieren des Texts beschädigt worden. Die korrekte Kopfzeile habe ich angehängt. Antworte ausschließlich mit einer Liste im Format [zahl_1, zahl_2, zahl_3], die ausschließlich alle die genannten Prozentzahlen enthält, die sich auf den Prozentsatz des auf der Hauptversammlung vertretenen Grundkapitals beziehen. Durchsuche das ganze Dokument nach solchen Zahlen, auch außerhalb von Tabellen. Wenn du dir nicht absolut sicher bist, antworte mit [0]. Verwende Punkt statt Komma für die Zahlen. \n Korrekte Kopfzeile: "
        print(user_prompt)
        user_prompt += column_names + "\n Bericht der Hauptversammlung: "

        response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
        header_row_list.append(response)
        data['cost'] += cost

    try:
        # Parse percentage list from response for the entire document
        percentage_list = ast.literal_eval(response)
        if percentage_list:
            highest_percentage = max(percentage_list)
            data['Presence_predicted'] = round(highest_percentage, 2)
    except (SyntaxError, ValueError) as e:
        data["comment"] += f"{response} "
        data.update({
            "error": "Error parsing list of percentages"
        })
        log_to_wandb(data)
        continue

    # sanity checks

    if len(highest_percentage_list) > 1:
        second_highest_percentage = sorted(highest_percentage_list, reverse=True)[1]
        if highest_percentage - second_highest_percentage > 10:
            data["comment"] += "Der ermittelte Wert weicht um mehr als 10 Prozentpunkte vom zweithöchsten Wert ab. "
            data.update({
                "error": "Significant difference from second highest value",
                "Presence_predicted": highest_percentage
            })
            log_to_wandb(data)
            continue

    print("highest_percentage: ", data['Presence_predicted'])

    if data['Presence_enhanced'] == data['Presence_predicted']:
        data['correct'] = True
    else:
        # Check for another row with the same ID_Key_original and Year_original
        same_id_year_rows = df[(df['ID_Key_original'] == row['ID_Key_original']) & 
                            (df['Year_original'] == row['Year_original'])]

        # Check if any of those rows have Presence_enhanced equal to highest_percentage
        if any(same_id_year_rows['Presence_enhanced'] == data['Presence_predicted']):
            data['correct'] = True
            data["comment"] += "Der ermittelte Wert stammt aus dem anderen Bericht diesen Jahres und ist dort korrekt ermittelt. "
        else:
            data['correct'] = False

    # Log the data
    log_to_wandb(data)

print("system_prompt: ", system_prompt)
print("user_prompt: ", user_prompt)

# Save the table to W&B
wandb.log({"results": table})