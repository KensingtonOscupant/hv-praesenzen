from tqdm import tqdm
import ast
import re
from collections import Counter
import pandas as pd
from openai import OpenAI
import wandb
import json
from config import args, api_key
from utilities import calculate_cost, find_subdirectory, find_matching_files, process_pdf, call_language_model

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

run = wandb.init(project=config['wandb_project_name'])
# Define W&B Table to store results
columns = config['wandb_table_columns']
table = wandb.Table(columns=columns)
# log a row of data to the W&B table
def log_to_wandb(data):
    table.add_data(*data.values())

# Log the config.json file as an artifact
artifact = wandb.Artifact('run_configurations', type='config')
artifact.add_file('config.json')
run.log_artifact(artifact)

# Assuming args.start and args.end are defined (the row numbers to start and end with)
start_index = args.start if args.start is not None else 0
end_index = args.end if args.end is not None else len(df)

# Calculate the total number of rows to be processed
total_rows = end_index - start_index

# Main application logic
if __name__ == "__main__":
    # Use args and api_key as needed
    print(f"Start: {args.start}, End: {args.end}")

    # keep track of processed files
    processed_files = {}

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
            "header_row": None,
            "all_percentage_values": None,
            "file_path": None,
            "comment": ""
        }

        # set flag for alternative document structure, i.e., no table but more like a list
        alternative_document_structure = False

        id_value = str(int(row['ID_Key_original']))
        year_suffix = str(int(row['Year_original']))[-2:]

        # find the subdirectory where all reports for the same ID are located
        subdirectory_path = find_subdirectory(directory, id_value)

        # find the file path for the report(s) of the given year
        matching_files = find_matching_files(subdirectory_path, year_suffix)

        # Initialize the key in the dictionary if it doesn't exist
        key = (id_value, year_suffix)
        if key not in processed_files:
            processed_files[key] = []

        # Find the first unprocessed file
        unprocessed_files = [file for file in matching_files if file not in processed_files[key]]
        if not unprocessed_files:
            # All files for this ID and year have been processed
            continue

        # Process the first unprocessed file
        file_path = unprocessed_files[0]

        # mark the file as processed
        processed_files[key].append(file_path)

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

        system_prompt = config['prompts']['system_prompt']
        user_prompt = config['prompts']['user_prompts']['header_evaluation_prompt'].format(report=full_text)

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
            data['header_row'] = most_common_header
        elif len(most_common_headers) > 1:
            # Multiple strings with the same highest frequency
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

            # check if response is 0. In that case, we check if the file might not be structured like a table, but more like a list (alternative document structure)
            if column_names == "[0]":

                user_prompt = config['prompts']['user_prompts']['alternative_doc_structure_prompt'].format(report=full_text)

                response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
                header_row_list.append(response)
                data['cost'] += cost

                # check response
                if response == "[1]":

                    alternative_document_structure = True

                    # find the highest grundkapital percentage in the document
                    user_prompt = config['prompts']['user_prompts']['alternative_doc_extraction_prompt'].format(report=full_text)

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
            user_prompt = config['prompts']['user_prompts']['regular_extraction_prompt'].format(column_names=column_names, report=full_text)

            response, cost = call_language_model(client, system_prompt, user_prompt, full_text, calculate_cost)
            header_row_list.append(response)
            data['cost'] += cost

        try:
            # Parse percentage list from response for the entire document
            percentage_list = ast.literal_eval(response)
            if percentage_list:
                highest_percentage = max(percentage_list)
                data['Presence_predicted'] = round(highest_percentage, 2)
                data['all_percentage_values'] = percentage_list
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

    # Save the table to W&B
    wandb.log({"results": table})

    wandb.finish()