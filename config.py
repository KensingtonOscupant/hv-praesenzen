import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")