import weave
from weave import Model
from openai import OpenAI
import os
from pydantic import BaseModel
import asyncio
import json
from utilities import process_pdf
from dotenv import load_dotenv

load_dotenv()

# Load the config file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

config = load_config('config.json')

client = weave.init("agm_share_capital_present_13")

llm_name = os.getenv("LLM_NAME")
split = os.getenv("SPLIT")
prompt_name = os.getenv("PROMPT_NAME")

# create prompt objects
system_prompt = weave.StringPrompt(config['prompts']['system_prompt'])
single_label_present_prompt = weave.StringPrompt(config['prompts']['user_prompts']['single_label_present_prompt'])
single_label_value_prompt = weave.StringPrompt(config['prompts']['user_prompts']['single_label_value_prompt'])

#publish for version control
weave.publish(system_prompt, name="system_prompt")
weave.publish(single_label_value_prompt, name="single_label_value_prompt")
weave.publish(single_label_present_prompt, name="single_label_present_prompt")

class LabelValueOutput(BaseModel):
    label_value_llm_output: float

class AGMPresenceModel(Model):

    @weave.op()
    def predict(self, file_path: str):
        # TODO get metadata, specifically year, id_key, maybe unique id
        metadata = "metadata"
        text, error = process_pdf(file_path)  # Extract text from PDF

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Then try regular extraction with the header info
        label_value_response = client.responses.parse(
            model=llm_name, 
            input=[
                {"role": "system", "content": system_prompt.content},
                {"role": "user", "content": single_label_value_prompt.content.format(
                    report=text
                )}
            ],
            text_format=LabelValueOutput
        )

        return {'label_value_predicted': round(label_value_response.output_parsed.label_value_llm_output, 2),
                'metadata': metadata}

model = AGMPresenceModel(name="single_prompt_o3_mini", description="Plain attempt with just a single prompt, no few-shot examples.")

eval = weave.ref(f"{split}_eval").get()

asyncio.run(eval.evaluate(model))

weave.finish()