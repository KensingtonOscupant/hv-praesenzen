import weave
from weave import Model
from openai import OpenAI
import os
from pydantic import BaseModel
import asyncio
import json
from utilities import process_pdf, get_prompt
from openai_prices import price 
from dotenv import load_dotenv

load_dotenv()

client = weave.init("agm_minimum_share_capital_present_15")

llm_name = os.getenv("LLM_NAME")
price_per_1m_tokens = price(llm_name)

split = os.getenv("SPLIT")
prompt_name = os.getenv("PROMPT_NAME")
description = os.getenv("MODEL_DESCRIPTION")

# name of the run that appears on the wandb leaderboard
leaderboard_name = f"{llm_name}_using_{prompt_name}"

system_prompt = get_prompt("system_prompt")
base_prompt = get_prompt("base_prompt")
header_row_prompt = get_prompt("header_row_prompt")

class LabelValueOutput(BaseModel):
    label_value_llm_output: float

class HeaderRowOutput(BaseModel):
    header_row_output_llm_output: bool

class AGMPresenceModel(Model):
    """Keeping these here only because it enables automatic prompt versioning in Weave.
    Due to a deserialization issue, it is not possible to reference these as class attributes
    further down, which is why I am falling back to using the corresponding global variables."""
    system_prompt: weave.trace.refs.ObjectRef
    base_prompt: weave.trace.refs.ObjectRef
    header_row_prompt: weave.trace.refs.ObjectRef

    @weave.op()
    def predict(self, file_path: str):
        # TODO get metadata, specifically year, id_key, maybe unique id
        metadata = "metadata"
        text, error = process_pdf(file_path)  # Extract text from PDF

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        header_row_present_response = client.responses.parse(
            model=llm_name, 
            input=[
                {"role": "system", "content": system_prompt.get().content},
                {"role": "user", "content": header_row_prompt.get().content.format(
                    report=text
                )}
            ],
            text_format=HeaderRowOutput
        )

        # Then try regular extraction with the header info
        label_value_response = client.responses.parse(
            model=llm_name, 
            input=[
                {"role": "system", "content": system_prompt.get().content},
                {"role": "user", "content": base_prompt.get().content.format(
                    report=text
                )}
            ],
            text_format=LabelValueOutput
        )

        # helper for cost tracking; currently doesn't work with this setup
        # due to wandb-acknowledged bug
        u = label_value_response.usage
        cost = (u.input_tokens * price_per_1m_tokens["input"] / 1_000_000
                + u.output_tokens * (price_per_1m_tokens["output"] / 1_000_000))

        if label_value_response.output_parsed.label_value_llm_output == -2:
            label_value_predicted = None
        else:     
            label_value_predicted = round(label_value_response.output_parsed.label_value_llm_output, 2)

        return {'label_value_predicted': label_value_predicted,
                'header_row_present': header_row_present_response.output_parsed.header_row_output_llm_output,
                'metadata': metadata,
                'cost': cost}

model = AGMPresenceModel(name=leaderboard_name, description=description, system_prompt=system_prompt, base_prompt=base_prompt, header_row_prompt=header_row_prompt)

eval = weave.ref(f"{split}_eval").get()

asyncio.run(eval.evaluate(model))

weave.finish()