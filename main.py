

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from engine import SmartSwapEngine

app = FastAPI(title="SmartSwap Decision Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SmartSwapEngine()

class RecommendRequest(BaseModel):
    current_tool: str
    team_size: int = 10
    max_budget_monthly: float = 50.0
    must_have_features: List[str] = []
    natural_query: Optional[str] = ""


# In main.py:

@app.get("/")
def serve_login():
    """Serves the login / registration gateway first."""
    if os.path.exists("login.html"):
        return FileResponse("login.html")
    return FileResponse("index.html")

@app.get("/dashboard")
def serve_dashboard():
    """Serves the main SmartSwap recommender engine workspace."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html not found"}

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html not found"}

@app.get("/tools")
def get_tools():
    tools = engine.df[["tool_name", "category", "license_type"]].to_dict(orient="records")
    return {"tools": tools}

@app.get("/features")
def get_features():
    features = [
        {
            "id": col,
            "label": col.replace("has_", "").replace("is_", "").replace("_", " ").title()
        }
        for col in engine.feature_cols
    ]
    return {"features": features}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    result = engine.recommend(
        current_tool_name=req.current_tool,
        team_size=req.team_size,
        max_budget_monthly=req.max_budget_monthly,
        must_have_features=req.must_have_features,
        natural_query=req.natural_query or ""
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)