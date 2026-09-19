"""Twin Model Experiment Runner: Evaluates dose-controlled models (0%, 25%, 50%, 75%, 100%)

across Task Competence (C), Temporal Leakage (L), and Leakage-Induced Economic Effect (E_L).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .fomc_benchmark import FOMCBenchmark
from .metrics import (
    evaluate_behavioral_leakage,
    evaluate_competence,
    evaluate_economic_effect,
    evaluate_representational_leakage,
    pareto_coordinates,
)
from .temporal_model import MockTwinEncoder, NullConstantModel, TemporalModel


class TwinExperimentRunner:
    """Orchestrates controlled twin model evaluation across contamination doses."""

    def __init__(
        self,
        benchmark: Optional[FOMCBenchmark] = None,
        output_dir: str = "experiments/fomc_temporal_leakage",
    ) -> None:
        self.benchmark = benchmark or FOMCBenchmark()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        models: Optional[Dict[str, TemporalModel]] = None,
    ) -> pd.DataFrame:
        """Run complete evaluation across all twin models on the out-of-sample test split.

        Returns:
            pd.DataFrame summarizing C, L_repr, L_mask, E_L, and 3D Pareto coordinates.
        """
        test_samples = self.benchmark.get_split("test")
        texts = [s.text for s in test_samples]
        y_true = [s.task_label for s in test_samples]
        # Economic targets: forward 5-day return and next meeting action
        fwd_returns = [s.market_outcomes.get("spy_5d_return", 0.0) for s in test_samples]
        y_future_actions = [s.future_macro_labels.get("next_action", 0) for s in test_samples]

        # Default twin ladder if not provided
        if models is None:
            models = {
                "M_clean": MockTwinEncoder("M_clean", competence_level=0.75, leakage_dose=0.0),
                "M_leak_25": MockTwinEncoder("M_leak_25", competence_level=0.75, leakage_dose=0.25),
                "M_leak_50": MockTwinEncoder("M_leak_50", competence_level=0.75, leakage_dose=0.50),
                "M_leak_75": MockTwinEncoder("M_leak_75", competence_level=0.75, leakage_dose=0.75),
                "M_leak_100": MockTwinEncoder("M_leak_100", competence_level=0.75, leakage_dose=1.00),
                "M_0_null": NullConstantModel(constant_class=0),
            }

        # Reference clean embeddings & stances for differential evaluation
        clean_model = models.get("M_clean") or list(models.values())[0]
        h_clean = clean_model.encode(texts)
        s_clean = clean_model.get_stance_score(texts)

        results = []
        for name, model in models.items():
            # 1. Competence C
            preds, probs = model.predict_task(texts)
            comp_res = evaluate_competence(y_true, preds, y_prob=probs, n_bootstrap=300)

            # 2. Representational Leakage L_repr
            h_model = model.encode(texts)
            repr_res = evaluate_representational_leakage(h_model, h_clean, y_future_actions, n_permutations=200)

            # 3. Behavioral Leakage L_mask
            behav_res = evaluate_behavioral_leakage(model, test_samples)

            # 4. Economic Effect E_L
            s_model = model.get_stance_score(texts)
            econ_res = evaluate_economic_effect(s_model, s_clean, fwd_returns)

            # Composite L (average of representational and total masking leakage)
            l_composite = float(max(0.0, repr_res["l_repr"]) * 0.5 + behav_res["l_total_mask"] * 0.5)
            # For clean controls and null models with zero contamination, leakage-induced alpha is 0.0
            if model.contamination_dose == 0.0 or isinstance(model, NullConstantModel):
                e_l = 0.0
            else:
                e_l = float(econ_res["delta_sharpe"])
            c_val = comp_res["macro_f1"]

            p_coords = pareto_coordinates(c=c_val, l=l_composite, e_l=e_l)

            row = {
                "model_name": name,
                "dose": model.contamination_dose,
                "competence_macro_f1": c_val,
                "competence_mcc": comp_res["mcc"],
                "l_repr": repr_res["l_repr"],
                "l_entity": behav_res["l_entity"],
                "l_date": behav_res["l_date"],
                "l_total_mask": behav_res["l_total_mask"],
                "l_composite": l_composite,
                "e_l_delta_sharpe": e_l,
                "e_l_delta_ic": econ_res["delta_ic"],
                "e_l_delta_return": econ_res["delta_annual_return"],
                "p_value_sharpe": econ_res["p_value_sharpe"],
                "pareto_L": p_coords[0],
                "pareto_EL": p_coords[1],
                "pareto_C": p_coords[2],
            }
            results.append(row)

        df = pd.DataFrame(results)

        # Compute marginal leakage return dE_L / dL for the dose ladder
        df_ladder = df[df["model_name"].str.startswith("M_leak") | (df["model_name"] == "M_clean")].sort_values("dose")
        if len(df_ladder) > 1:
            dl = df_ladder["l_composite"].diff().replace(0, 1e-6)
            de = df_ladder["e_l_delta_sharpe"].diff()
            df_ladder["marginal_dEL_dL"] = de / dl
            df = df.merge(df_ladder[["model_name", "marginal_dEL_dL"]], on="model_name", how="left")
        else:
            df["marginal_dEL_dL"] = 0.0

        # Save outputs
        df.to_csv(self.output_dir / "twin_experiment_results.csv", index=False)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

        return df
