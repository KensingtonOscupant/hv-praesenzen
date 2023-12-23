# Determining the minimum base capital present at Annual Shareholders' Meetings (ASMs) based on ASM reports with GPT-4

## Description
This project evaluates the minimum base capital in Annual Shareholder Meetings (ASMs) of German companies, achieving 98.4% accuracy. The test set, representing a comprehensive sample, includes one report from each of the 178 companies assessed, drawn from a pool of approximately 3700 reports.

## Features
- Processing PDF reports to extract relevant data.
- Using ```gpt-4-1106-preview``` for interpreting and analyzing textual data.
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

The script functions based on settings defined in a 'config.json' file. This file allows users to specify various parameters, including the path to the evaluated reports, the location for storing results, and options for inputting a test set. It also includes settings for the Weights & Biases (W&B) project and, crucially, the prompts utilized. Additionally, the 'config.json' file is recorded as an artifact in W&B, simplifying the process of tracking the settings for each run.

## Usage

To run the script, use the following command:

```python main.py```

You can supply the command line arguments --start [your start row] and --end [your last row] if you only want to run it on a slice of your dataframe, e.g. ```python main.py --start 2 --end 30``` to only evaluate rows 2 to 30 of your dataframe.
