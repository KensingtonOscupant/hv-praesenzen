import weave
import os
import asyncio
from utilities import get_prompt
from dotenv import load_dotenv


load_dotenv()

llm_name         = os.getenv("LLM_NAME")
split            = os.getenv("SPLIT")
prompt_setup     = os.getenv("PROMPT_NAME") 
description      = os.getenv("MODEL_DESCRIPTION")
project_name     = os.getenv("PROJECT_NAME")

client           = weave.init(project_name)
from model import AGMPresenceModel
leaderboard_name = f"{llm_name}_using_{prompt_setup}"

system_prompt    = get_prompt("system_prompt")
base_prompt      = get_prompt("base_prompt")

model = AGMPresenceModel(name=leaderboard_name, description=description, system_prompt=system_prompt, base_prompt=base_prompt)
eval = weave.ref(f"{split}_eval").get()
asyncio.run(eval.evaluate(model))

weave.finish()