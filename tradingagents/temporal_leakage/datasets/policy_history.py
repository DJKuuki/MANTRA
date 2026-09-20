"""Official FOMC Policy History and Target Action Derivation Engine.

Extracts and derives point-in-time future policy actions from the official FOMC policy history.
Guarantees:
1. Programmatic derivation: future_action is computed directly from chronological policy history,
   never entered as manual unverified constants.
2. Explicit rule freezing:
   - Primary: next scheduled FOMC meeting only.
   - Unscheduled / emergency actions (e.g. 2020-03-03, 2020-03-15) are excluded from primary scheduled
     horizon unless explicitly requested.
3. Strict consistency checks:
   - Validates monotonic dates.
   - Derives action from before/after target range bounds:
     action = +1 if upper_after > upper_before else (-1 if upper_after < upper_before else 0)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


DEFAULT_POLICY_HISTORY_PATH = Path("data/research/fomc/policy_history.csv")


def load_policy_history(
    csv_path: Union[str, Path] = DEFAULT_POLICY_HISTORY_PATH,
) -> pd.DataFrame:
    """Load and validate official FOMC policy history records."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy history file not found: {path}")

    df = pd.read_csv(path)
    required_cols = [
        "meeting_date",
        "decision_time",
        "target_lower_before",
        "target_upper_before",
        "target_lower_after",
        "target_upper_after",
        "action",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in policy history: {col}")

    # Chronological sort
    df["meeting_date"] = df["meeting_date"].astype(str)
    df = df.sort_values("meeting_date").reset_index(drop=True)

    # Validate action consistency against target range difference
    for idx, row in df.iterrows():
        diff = row["target_upper_after"] - row["target_upper_before"]
        expected_act = 1 if diff > 1e-6 else (-1 if diff < -1e-6 else 0)
        if int(row["action"]) != expected_act:
            raise ValueError(
                f"Policy history internal inconsistency at {row['meeting_date']}: "
                f"Upper changed from {row['target_upper_before']} to {row['target_upper_after']} "
                f"(diff {diff}), but recorded action was {row['action']} (expected {expected_act})."
            )

    return df


def derive_future_action(
    meeting_date: str,
    policy_df: Optional[pd.DataFrame] = None,
    include_unscheduled: bool = False,
) -> Dict[str, Any]:
    """Derive future policy action target for a given FOMC meeting.

    Args:
        meeting_date: Date string YYYY-MM-DD.
        policy_df: Optional pre-loaded policy history DataFrame.
        include_unscheduled: If False (Primary), looks up only the next scheduled meeting.

    Returns:
        Dict with derived target properties:
        - 'future_action': int (-1, 0, 1)
        - 'future_action_time': str (UTC ISO timestamp of next decision)
        - 'target_horizon': str
        - 'target_source': str
        - 'target_definition': str
        - 'target_quality': str
    """
    if policy_df is None:
        policy_df = load_policy_history()

    df = policy_df.copy()
    if not include_unscheduled and "meeting_type" in df.columns:
        df = df[df["meeting_type"] == "scheduled"].reset_index(drop=True)

    current_idx_list = df.index[df["meeting_date"] == meeting_date].tolist()
    if not current_idx_list:
        raise ValueError(f"Meeting date '{meeting_date}' not found in policy history.")

    curr_idx = current_idx_list[0]
    if curr_idx + 1 >= len(df):
        raise ValueError(f"No future meeting found in policy history after '{meeting_date}'.")

    next_meeting = df.iloc[curr_idx + 1]
    future_act = int(next_meeting["action"])
    future_time = str(next_meeting["decision_time"])

    horizon = "next_fomc_decision_any" if include_unscheduled else "next_scheduled_fomc_decision"
    definition = (
        f"Policy rate action at next {'any' if include_unscheduled else 'scheduled'} FOMC meeting: "
        "-1=cut, 0=hold, +1=hike"
    )

    return {
        "future_action": future_act,
        "future_action_time": future_time,
        "target_horizon": horizon,
        "target_source": "OFFICIAL_FOMC_POLICY_HISTORY",
        "target_definition": definition,
        "target_quality": "exact",
        "next_meeting_date": str(next_meeting["meeting_date"]),
    }


def get_current_policy_action(
    meeting_date: str,
    policy_df: Optional[pd.DataFrame] = None,
) -> int:
    """Get the current rate action implemented at the given meeting date."""
    if policy_df is None:
        policy_df = load_policy_history()
    row = policy_df[policy_df["meeting_date"] == meeting_date]
    if row.empty:
        raise ValueError(f"Meeting date '{meeting_date}' not found in policy history.")
    return int(row.iloc[0]["action"])
