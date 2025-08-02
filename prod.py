import weave
import os
from dotenv import load_dotenv
import glob

load_dotenv()

project_name = os.getenv("PROJECT_NAME")
client = weave.init(project_name)

from utilities import get_prompt

from model import AGMPresenceModel

llm_name = os.getenv("LLM_NAME")
prompt_setup = os.getenv("PROMPT_SETUP") 
description = os.getenv("MODEL_DESCRIPTION")

leaderboard_name = f"{llm_name}_using_{prompt_setup}"

system_prompt = get_prompt("system_prompt")
base_prompt = get_prompt("base_prompt")

model = AGMPresenceModel(name=leaderboard_name, description=description, system_prompt=system_prompt, base_prompt=base_prompt)

# Process all PDF files in the data directory
data_directory = "data/Praesenzen_hv-info"
file_pattern = os.path.join(data_directory, "**", "*.pdf")
file_path_list = glob.glob(file_pattern, recursive=True)

for file_path in file_path_list:
    
    try:
        prediction_result = model.predict(file_path=file_path)
    except Exception as e:
        print(f"Error making prediction for {file_path}: {str(e)}")

weave.finish()