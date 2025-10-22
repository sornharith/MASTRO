"""
Prompt-based agent implementation for the multi-agent dropout prediction system
"""
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils.logger import log
from llm.utils import safe_cast, parse_json_from_response


class PromptRiskMixin:
    """
    Mixin class for prompt-based agents that use LLMs for risk assessment and rationale generation
    """
    def __init__(self, name, risk_fn, rat_fn, llm, json_extractor_chain=None):
        self.name = name
        self.risk_fn = risk_fn
        self.rat_fn = rat_fn
        self.llm = llm
        self.json_extractor_chain = json_extractor_chain
        self.chain = None
        
        # Set up the chain if LLM is provided
        if llm:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # We'll set this up when the prompt is provided
            pass

    def _snap_stage_feats(self, row):
        snap = int(row.get("snapshot_day", 30))
        stage = "early" if snap <= 90 else "mid" if snap <= 180 else "late"
        
        # Safely get column names for the feature summary
        top_cols = []
        if hasattr(self.rat_fn, '__self__') and hasattr(self.rat_fn.__self__, 'cols'):
            top_cols = self.rat_fn.__self__.cols
        else:
            top_cols = row.index[:3]
            
        feats = ", ".join(f"{k}={row[k]:.3g}" for k in top_cols if k in row)
        return snap, stage, feats

    def _extras(self, row, p_num, cluster_stats=None):
        if self.name == "CourseDifficulty":
            return {"cluster_info": str(cluster_stats), "numeric_risk": f"{p_num:.3f}"}
        elif self.name == "Engagement":
            return {"clicks_total": f"{row.get('clicks_total',0):.0f}"}
        elif self.name == "Temporal":
            return {
                "days_since_last_activity": int(row.get("days_since_last_activity",0)),
                "clicks_recent_vs_early":  f"{row.get('clicks_recent_vs_early',0):.2f}"
            }
        return {}

    def predict(self, X, cluster_stats=None):
        preds, rats = [], []

        for _, row in tqdm(X.iterrows(), total=len(X), unit="rec",
                        desc=f"→ {self.name} predict"):
            p_base = float(self.risk_fn(row))
            
            # If there's no LLM, just use the base deterministic prediction
            if not self.llm or not self.json_extractor_chain:
                preds.append(p_base)
                rats.append(self.rat_fn(row))
                continue

            # Build the prompt payload for the LLM
            snap, stage, feats = self._snap_stage_feats(row)
            payload = {
                "snapshot_day": snap, "stage": stage, "numeric_risk": f"{p_base:.3f}",
                "top_feats": feats, "features_json": row.to_json(),
                "cluster_info": "-", "peer_delta": "-", "gate": "-", "clicks_total": "-",
                "days_since_last_activity": "-", "clicks_recent_vs_early": "-"
            }
            payload.update(self._extras(row, p_base, cluster_stats))

            # Run the LLM chain
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            # Create chain for this specific agent if not exists
            if not self.chain:
                from llm.prompts import PROMPT_TS
                self.chain = LLMChain(
                    llm=self.llm,
                    prompt=PromptTemplate.from_template(
                        PROMPT_TS[self.name], template_format="jinja2"
                    )
                )
                
            raw = self.chain.run(payload)
            
            # Handle the special case for the CourseDifficulty agent
            if self.name == "CourseDifficulty":
                preds.append(p_base)
                rats.append(raw.strip())
                continue

            # --- Robust Parsing Block ---
            # Attempt to parse the JSON from the LLM response
            parsed = parse_json_from_response(self.json_extractor_chain.run(raw_text=raw))
            
            # Safely get the 'risk' value, defaulting to the base prediction
            risk_value = parsed.get("risk", p_base)
            
            # Safely cast the risk value to a float, defaulting again if it fails
            numeric_risk = safe_cast(risk_value, float, p_base)
            
            preds.append(numeric_risk)
            rats.append(parsed.get("rationale", "Rationale could not be parsed."))

        return np.array(preds), rats