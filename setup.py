import pandas as pd
import weave
from weave import Dataset, Evaluation, Scorer
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
from typing import Any
from agm_scorer import AGMPresenceScorer

weave.init("agm_share_capital_present_13")

for split in ["train", "dev", "test"]:

    # upload dataset
    csv_path = f"data/250907_{split}_set.csv"
    df = pd.read_csv(csv_path)
    dataset_object = Dataset.from_pandas(df)
    dataset_ref = weave.publish(dataset_object, name=f"{split}_set") # at least i think it's a ref no?

    # create eval for respective split
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
            summary_metric_path="value.accuracy",
                )
            ],
    )

    weave.publish(leaderboard_spec, f"{split}_set_leaderboard")
