# Money‑Printer Screener


A Streamlit app to find high‑quality, high‑FCF companies that *might* be undervalued.


## Why this exists
- "Money printer" ≈ strong, repeatable **free cash flow** (FCF)
- Undervaluation proxies: **FCF yield (EV), EV/EBIT**, margins, growth, leverage, and buybacks
- 100% transparent math you can tweak


## Quickstart
```bash
# 1) Clone your new repo
# (create a GitHub repo and add these 3 files: app.py, requirements.txt, README.md)


# 2) Create & activate a virtual env
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate


# 3) Install deps
pip install -r requirements.txt


# 4) Run the app
streamlit run app.py
