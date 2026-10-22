# ⇄ SmartSwap: SaaS Alternative & Cost Optimization Engine

SmartSwap is a multi-attribute utility recommendation system built to eliminate enterprise software budget waste. Instead of sponsored directories, it utilizes constraint satisfaction and vector distance scoring to identify functional, budget-friendly commercial and open-source alternatives.

---

## ⚡ Core Features

* **Multi-Criteria Optimization:** Calculates recommendation rankings using a hybrid of feature cosine similarity and NLP text similarity (TF-IDF).
* **Deterministic Constraint Filtering:** Eliminates tools that fail must-have features or exceed hard budget bounds.
* **Feature Diffing & ROI Math:** Dynamically calculates annual cost savings alongside retained capabilities vs. sacrificed trade-offs.
* **Reactive Light-Mode Interface:** Pure modern frontend with zero build-step overhead, providing instant debounced updates.

---

## 🛠 Tech Stack

* **Backend:** Python 3.12, FastAPI, Pydantic, Uvicorn
* **ML / NLP:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity), NumPy, Pandas
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+)

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/EshitaG05/smart-swap.git](https://github.com/yourusername/smart-swap.git)
   cd smart-swap