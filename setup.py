import pandas as pd
import weave
from weave import Dataset, Evaluation, Scorer
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
from typing import Any
from scorer import AGMPresenceScorer
from dotenv import load_dotenv
import os

load_dotenv()

project_name = os.getenv("PROJECT_NAME")
dataset_name = os.getenv("DATASET_NAME")

weave.init(project_name)

for split in ["train", "dev", "test"]:

    # upload dataset
    csv_path = f"data/{dataset_name}_{split}_set.csv"
    df = pd.read_csv(csv_path)
    dataset_object = Dataset.from_pandas(df)
    dataset_ref = weave.publish(dataset_object, name=f"{split}_set")

    if split == "test":
        # create eval object for test
        evaluation = Evaluation(
            dataset=dataset_ref,
            scorers=[AGMPresenceScorer()],
            trials=10
        )

    else:
        # create eval objects for train and dev
        evaluation = Evaluation(
            dataset=dataset_ref, 
            scorers=[AGMPresenceScorer()],
        )


    weave.publish(evaluation, name=f"{split}_eval")

    # create leaderboard for respective split
    leaderboard_spec = leaderboard.Leaderboard(
        columns=[
            leaderboard.LeaderboardColumn(
            evaluation_object_ref=get_ref(evaluation).uri(),
            scorer_name="AGMPresenceScorer",
            summary_metric_path="accuracy",
                ),
            leaderboard.LeaderboardColumn(
            evaluation_object_ref=get_ref(evaluation).uri(),
            scorer_name="AGMPresenceScorer",
            summary_metric_path="selective_accuracy"
            ),
            leaderboard.LeaderboardColumn(
            evaluation_object_ref=get_ref(evaluation).uri(),
            scorer_name="AGMPresenceScorer",
            summary_metric_path="coverage"
            )
            ],
    )

    weave.publish(leaderboard_spec, f"{split}_set_leaderboard")
