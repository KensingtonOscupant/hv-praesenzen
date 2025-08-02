import os
import weave
from pydantic import BaseModel
from openai import OpenAI
from weave import Model
from dotenv import load_dotenv
from utilities import process_pdf, get_prompt, get_metadata

load_dotenv()

llm_name         = os.getenv("LLM_NAME")

system_prompt    = get_prompt("system_prompt")
base_prompt      = get_prompt("base_prompt")

class LabelValueOutput(BaseModel):
    label_value_llm_output: list[float]

class AGMPresenceModel(Model):
    """Keeping these here only because it enables automatic prompt versioning in Weave.
    Due to a deserialization issue, it is not possible to reference these as class attributes
    further down, which is why I am falling back to using the corresponding global variables."""
    system_prompt: weave.trace.refs.ObjectRef
    base_prompt: weave.trace.refs.ObjectRef

    @weave.op()
    def predict(self, file_path: str):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        metadata = get_metadata(file_path)
        pages, error = process_pdf(file_path)
        
        page_analysis_results = []
        for page in pages:

            # Then try regular extraction with the header info
            label_value_response = client.responses.parse(
                model=llm_name, 
                input=[
                    {"role": "system", "content": system_prompt.get().content},
                    {"role": "user", "content": base_prompt.get().content.format(
                        report=page
                    )}
                ],
                text_format=LabelValueOutput
            )

            page_analysis_results.extend(label_value_response.output_parsed.label_value_llm_output)

        print(f"{page_analysis_results}")
        # Get the highest value from all page analysis results
        highest_value = max(page_analysis_results) if page_analysis_results else -2

        return {'label_value_predicted': round(highest_value, 2),
                'metadata': metadata,
                'error': error
                }