

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class SmartSwapEngine:
    def __init__(self, data_path: str = "saas_tools.csv"):
        self.df = pd.read_csv(data_path)
        
        # Binary feature columns
        self.feature_cols = [
            col for col in self.df.columns 
            if col.startswith("has_") or col.startswith("is_")
        ]
        
        # Lightweight NLP: TF-IDF vectorizer over descriptions
        self.vectorizer = TfidfVectorizer(stop_words="english")
        descriptions = self.df["description"].fillna("").tolist()
        self.desc_matrix = self.vectorizer.fit_transform(descriptions)

    def calculate_monthly_cost(self, tool_row: pd.Series, team_size: int) -> float:
        effective_seats = max(team_size, int(tool_row["min_seats"]))
        return float(effective_seats * tool_row["base_price_user_month"])

    def estimate_migration_friction(self, current_tool: str, alt_tool: str, alt_license: str) -> dict:
        cur_lower = current_tool.lower()
        alt_lower = alt_tool.lower()
        
        # Automated importer heuristics
        if "jira" in cur_lower and ("plane" in alt_lower or "linear" in alt_lower):
            return {"level": "Low Effort", "badge_color": "emerald", "desc": "Direct Jira 1-click importer natively supported."}
        if alt_license == "Open Source":
            return {"level": "Moderate Effort", "badge_color": "amber", "desc": "Self-hosting and database provisioning required."}
        return {"level": "Low Effort", "badge_color": "emerald", "desc": "Direct CSV/JSON workspace migration supported."}

    def recommend(
        self,
        current_tool_name: str,
        team_size: int,
        max_budget_monthly: float,
        must_have_features: list[str] = None,
        natural_query: str = "",
        weight_features: float = 0.50,
        weight_semantic: float = 0.35,
        weight_cost: float = 0.15
    ):
        must_have_features = must_have_features or []
        
        # 1. Locate current tool
        cleaned_name = current_tool_name.strip().lower()
        current_matches = self.df[self.df["tool_name"].astype(str).str.strip().str.lower() == cleaned_name]

        if current_matches.empty:
            available = self.df["tool_name"].dropna().unique().tolist()
            return {
                "error": f"Tool '{current_tool_name}' not found. Available tools: {available}"
            }
            
        current_tool = current_matches.iloc[0]
        current_category = current_tool["category"]
        current_cost = self.calculate_monthly_cost(current_tool, team_size)
        current_features = current_tool[self.feature_cols].to_numpy(dtype=float)

        # 2. Filter candidate pool (same category, different tool, under budget, has must-haves)
        candidates = self.df[
            (self.df["category"] == current_category) & 
            (self.df["tool_name"].str.lower() != current_tool_name.lower())
        ].copy()

        candidates["calc_monthly_cost"] = candidates.apply(
            lambda r: self.calculate_monthly_cost(r, team_size), axis=1
        )
        
        # Budget filter
        candidates = candidates[candidates["calc_monthly_cost"] <= max_budget_monthly]

        # Hard must-have feature constraints
        for feature in must_have_features:
            if feature in self.feature_cols:
                candidates = candidates[candidates[feature] == 1]

        if candidates.empty:
            return {
                "message": "No direct matches meet your strict budget and mandatory constraints.",
                "suggestion": f"Try raising your budget above ${max_budget_monthly:.2f}/mo or unchecking some mandatory features."
            }

        # 3. Feature Cosine Similarity
        candidate_features = candidates[self.feature_cols].to_numpy(dtype=float)
        feat_sims = cosine_similarity(current_features.reshape(1, -1), candidate_features)[0]
        candidates["feature_score"] = np.nan_to_num(feat_sims, nan=0.0)

        # 4. Semantic Similarity via TF-IDF
        if natural_query.strip():
            query_vec = self.vectorizer.transform([natural_query])
            candidate_indices = candidates.index.tolist()
            subset_desc_matrix = self.desc_matrix[candidate_indices]
            semantic_scores = cosine_similarity(query_vec, subset_desc_matrix)[0]
            candidates["semantic_score"] = np.nan_to_num(semantic_scores, nan=0.0)
        else:
            candidates["semantic_score"] = candidates["feature_score"]

        # 5. Cost Optimization Score (higher savings ratio = higher bonus score)
        if current_cost > 0:
            candidates["cost_score"] = candidates["calc_monthly_cost"].apply(
                lambda c: max(0.0, min(1.0, (current_cost - c) / current_cost))
            )
        else:
            candidates["cost_score"] = 0.5

        # 6. Composite Ranking
        candidates["final_score"] = (
            (weight_features * candidates["feature_score"]) + 
            (weight_semantic * candidates["semantic_score"]) +
            (weight_cost * candidates["cost_score"])
        )

        ranked = candidates.sort_values(by="final_score", ascending=False).head(3)

        # 7. Build final payload
        results = []
        for _, alt in ranked.iterrows():
            alt_cost = alt["calc_monthly_cost"]
            savings_monthly = max(0.0, current_cost - alt_cost)
            savings_annual = savings_monthly * 12

            # Identify features kept, lost, and gained
            kept_features = [
                f.replace("has_", "").replace("is_", "").replace("_", " ").title()
                for f in self.feature_cols 
                if current_tool[f] == 1 and alt[f] == 1
            ]
            lost_features = [
                f.replace("has_", "").replace("is_", "").replace("_", " ").title()
                for f in self.feature_cols 
                if current_tool[f] == 1 and alt[f] == 0
            ]
            gained_features = [
                f.replace("has_", "").replace("is_", "").replace("_", " ").title()
                for f in self.feature_cols 
                if current_tool[f] == 0 and alt[f] == 1
            ]

            full_parity = {
                f.replace("has_", "").replace("is_", "").replace("_", " ").title(): {
                    "current": bool(current_tool[f]),
                    "candidate": bool(alt[f])
                }
                for f in self.feature_cols
            }

            migration_info = self.estimate_migration_friction(
                current_tool["tool_name"], 
                alt["tool_name"], 
                alt["license_type"]
            )

            results.append({
                "tool_name": alt["tool_name"],
                "license_type": alt["license_type"],
                "description": alt["description"],
                "match_score": round(float(alt["final_score"]) * 100, 1),
                "breakdown": {
                    "feature_score": round(float(alt["feature_score"]) * 100, 1),
                    "semantic_score": round(float(alt["semantic_score"]) * 100, 1),
                    "cost_score": round(float(alt["cost_score"]) * 100, 1),
                    "weights": {
                        "features": int(weight_features * 100),
                        "semantic": int(weight_semantic * 100),
                        "cost": int(weight_cost * 100)
                    }
                },
                "monthly_cost": f"${alt_cost:.2f}",
                "monthly_savings": f"${savings_monthly:.2f}",
                "annual_savings": f"${savings_annual:.2f}",
                "features_retained": kept_features,
                "trade_offs_lost": lost_features,
                "gained_features": gained_features,
                "migration_friction": migration_info,
                "full_parity": full_parity
            })

        return {
            "current_tool": current_tool["tool_name"],
            "current_category": current_category,
            "current_monthly_cost": f"${current_cost:.2f}",
            "team_size": team_size,
            "recommendations": results
        }