"""Twin Model Experiment Runner: Evaluates dose-controlled synthetic twins (0%, 25%, 50%, 75%, 100%)

across Task Competence (C), Representational Leakage (L_repr), Behavioral Leakage (L_behavior),
and Leakage-Induced Economic Effect (E_L).

Methodological Integrity:
1. No arbitrary composite leakage score (0.5 * L_repr + 0.5 * L_mask is strictly deleted).
2. Behavioral leakage is evaluated as Clean/Leak differential: Sensitivity(M_L) - Sensitivity(M_C).
3. Level A Information Coefficient (Delta IC) is the primary economic association metric.
4. Ground-truth future target is injected explicitly into the representation of contaminated twins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .fomc_benchmark import FOMCBenchmark, ToyFOMCBenchmark
from .metrics import (
    evaluate_behavioral_leakage,
    evaluate_competence,
    evaluate_economic_effect,
    evaluate_representational_leakage,
    pareto_coordinates,
)
from .temporal_model import NullConstantModel, SyntheticTemporalTwinEncoder, TemporalModel


class TwinExperimentRunner:
    """Orchestrates controlled twin model evaluation across contamination doses."""

    def __init__(
        self,
        benchmark: Optional[FOMCBenchmark] = None,
        output_dir: str = "experiments/fomc_temporal_leakage",
    ) -> None:
        self.benchmark = benchmark or ToyFOMCBenchmark()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        models: Optional[Dict[str, TemporalModel]] = None,
    ) -> pd.DataFrame:
        """Run complete evaluation across twin models on the out-of-sample test split.

        Returns:
            pd.DataFrame summarizing independent metrics for C, L_repr, L_behavior, and E_L.
        """
        test_samples = self.benchmark.get_split("test")
        texts = [s.text for s in test_samples]
        y_true = [s.task_label for s in test_samples]
        # Decoupled targets: future macro action for probing, forward returns for economics
        y_future_actions = [s.future_macro_labels.get("next_action", 0) for s in test_samples]
        fwd_returns = [s.market_outcomes.get("spy_5d_return", 0.0) for s in test_samples]

        # Default synthetic twin ladder if not provided
        if models is None:
            models = {
                "M_clean": SyntheticTemporalTwinEncoder("M_clean", competence_level=0.75, contamination_dose=0.0),
                "M_leak_25": SyntheticTemporalTwinEncoder("M_leak_25", competence_level=0.75, contamination_dose=0.25),
                "M_leak_50": SyntheticTemporalTwinEncoder("M_leak_50", competence_level=0.75, contamination_dose=0.50),
                "M_leak_75": SyntheticTemporalTwinEncoder("M_leak_75", competence_level=0.75, contamination_dose=0.75),
                "M_leak_100": SyntheticTemporalTwinEncoder("M_leak_100", competence_level=0.75, contamination_dose=1.00),
                "M_0_null": NullConstantModel(constant_class=0),
            }

        # Reference clean twin for differential estimation
        clean_model = models.get("M_clean") or list(models.values())[0]
        h_clean = clean_model.encode(texts, future_signals=None)
        s_clean = clean_model.get_stance_score(texts, future_signals=None)

        results = []
        for name, model in models.items():
            is_null = isinstance(model, NullConstantModel)
            future_sig = None if (model.contamination_dose == 0.0 or is_null) else y_future_actions

            # 1. Competence C
            preds, probs = model.predict_task(texts, future_signals=future_sig)
            comp_res = evaluate_competence(y_true, preds, y_prob=probs, n_bootstrap=200)

            # 2. Representational Leakage L_repr (TimeSeries expanding window & matched paired permutation)
            h_model = model.encode(texts, future_signals=future_sig)
            repr_res = evaluate_representational_leakage(
                h_model, h_clean, y_future_actions, probe_cv="timeseries", n_splits=3, n_permutations=100
            )

            # 3. Behavioral Leakage L_behavior (Clean/Leak Twin Differential)
            behav_res = evaluate_behavioral_leakage(
                model_leak=model,
                model_clean=clean_model,
                samples=test_samples,
                future_signals=future_sig,
            )

            # 4. Economic Effect E_L (Level A: Delta IC; Level B: Delta Sharpe)
            s_model = model.get_stance_score(texts, future_signals=future_sig)
            econ_res = evaluate_economic_effect(s_model, s_clean, fwd_returns)

            l_repr = 0.0 if (is_null or model.contamination_dose == 0.0) else repr_res["l_repr"]
            l_behavior = 0.0 if is_null else behav_res["l_behavior_delta"]
            l_entity = 0.0 if is_null else behav_res["l_entity_delta"]
            l_date = 0.0 if is_null else behav_res["l_date_delta"]

            e_l_ic = 0.0 if (is_null or model.contamination_dose == 0.0) else econ_res["delta_ic"]
            e_l_sharpe = 0.0 if (is_null or model.contamination_dose == 0.0) else econ_res["delta_sharpe"]
            e_l_return = 0.0 if (is_null or model.contamination_dose == 0.0) else econ_res["delta_annual_return"]

            c_val = comp_res["macro_f1"]
            mcc_val = comp_res["mcc"]

            p_coords = pareto_coordinates(
                c=c_val,
                l_repr=l_repr,
                l_behavior=l_behavior,
                e_l_ic=e_l_ic,
                e_l_sharpe=e_l_sharpe,
            )

            row = {
                "model_name": name,
                "dose": model.contamination_dose,
                "competence_macro_f1": c_val,
                "competence_mcc": mcc_val,
                "l_repr": l_repr,
                "l_behavior_delta": l_behavior,
                "l_entity_delta": l_entity,
                "l_date_delta": l_date,
                "mask_sensitivity_leak": behav_res["mask_sensitivity_leak"],
                "mask_sensitivity_clean": behav_res["mask_sensitivity_clean"],
                "e_l_delta_ic": e_l_ic,
                "e_l_delta_sharpe": e_l_sharpe,
                "e_l_delta_return": e_l_return,
                "p_value_ic": econ_res["p_value_ic"],
                "p_value_sharpe": econ_res["p_value_sharpe"],
                "pareto_C": p_coords["competence_C"],
                "pareto_L_repr": p_coords["leakage_L_repr"],
                "pareto_L_behavior": p_coords["leakage_L_behavior"],
                "pareto_EL_ic": p_coords["economic_E_L_ic"],
                "pareto_EL_sharpe": p_coords["economic_E_L_sharpe"],
            }
            results.append(row)

        df = pd.DataFrame(results)

        # Compute marginal leakage return dE_L / dL_repr across the dose ladder
        df_ladder = df[df["model_name"].str.startswith("M_leak") | (df["model_name"] == "M_clean")].sort_values("dose")
        if len(df_ladder) > 1:
            dl = df_ladder["l_repr"].diff().replace(0, 1e-6)
            de = df_ladder["e_l_delta_ic"].diff()
            df_ladder["marginal_dE_L_dL_repr"] = de / dl
            df = df.merge(df_ladder[["model_name", "marginal_dE_L_dL_repr"]], on="model_name", how="left")
        else:
            df["marginal_dE_L_dL_repr"] = 0.0

        # Save output records
        df.to_csv(self.output_dir / "twin_experiment_results.csv", index=False)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

        return df
