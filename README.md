# Shareholders' Meeting Minimum Base Capital Determination Script

## Description
This script is part of a project that analyzes the minimum base capital present in shareholders' meetings of various companies. It utilizes a range of Python libraries and APIs to process and analyze data from reports of shareholders' meetings (Hauptversammlungsberichte). The script has achieved a 98.4% accuracy on its test set.

## Features
- Processing PDF reports to extract relevant data.
- Using OpenAI's GPT models for interpreting and analyzing textual data.
- Integration with [Weights & Biases (wandb)](http://wandb.ai/) for experiment tracking and logging.
- Customized data handling and error logging mechanisms.

## Installation
Before running the script, ensure the following dependencies are installed:
- tqdm
- pandas
- openai
- wandb

You can install these packages using pip:
```pip install tqdm pandas openai wandb```

## Configuration

The script relies on a config.json file for various settings, including the path to the dataset (CSV file), OpenAI API key, and wandb project settings.

## Usage

To run the script, use the following command:

```python main.py```

You can supply the command line arguments --start [your start row] and --end [your last row] if you only want to run it on a slice of your dataframe, e.g. ```python main.py --start 2 --end 30``` to only evaluate rows 2 to 30 of your dataframe.
