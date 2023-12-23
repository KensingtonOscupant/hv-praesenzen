# Determining the minimum base capital present at Annual Shareholders' Meetings (ASMs) based on ASM reports with GPT-4

## Description
This project that analyzes the minimum base capital present in ASMs of German companies. The script has achieved a 98.4% accuracy on a test set sampled from at least one report of all the 178 companies evaluated.

## Features
- Processing PDF reports to extract relevant data.
- Using OpenAI's GPT models for interpreting and analyzing textual data.
- Integration with [Weights & Biases (wandb)](http://wandb.ai/) for experiment tracking and logging.

## Installation
Ensure the following dependencies are installed:
- tqdm
- pandas
- openai
- wandb
- pdfplumber
- python-dotenv

You can install these packages using pip:
```pip install tqdm pandas openai wandb pdfplumber python-dotenv```

To set up your OpenAI API key with dotenv, create a .env file in the base directory of the project and add ```OPENAI_API_KEY="sk-..."``` to it, where "sk-..." is your API key.

## Configuration

The script relies on a config.json file for various settings where one can set the path to the reports evaluated, the file where the results are stored (you can also have a test_set as input), W&B project settings, and most importantly, the prompts used. This file is also logged to W&B as an artifact, which makes it simple to keep track of the settings used in each run.

## Usage

To run the script, use the following command:

```python main.py```

You can supply the command line arguments --start [your start row] and --end [your last row] if you only want to run it on a slice of your dataframe, e.g. ```python main.py --start 2 --end 30``` to only evaluate rows 2 to 30 of your dataframe.
