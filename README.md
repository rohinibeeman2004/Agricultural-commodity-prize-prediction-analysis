# Agricultural Price Analytics - Streamlit

## Run

1. Open this folder in VS Code.
2. Open a terminal.
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Start Streamlit:

```bash
streamlit run app.py
```

5. Upload the agricultural price CSV/Excel file in the sidebar.

## Required columns

- t
- cmdty
- market_name
- district_name
- variety
- p_min
- p_max
- p_modal

## Included notebook workflow

Data loading, cleaning, EDA, feature engineering, correlation analysis,
Linear Regression, Ridge Regression, Lasso Regression, hyperparameter tuning,
Random Forest GridSearchCV, model comparison, feature importance, price
prediction and the notebook's RAG-style keyword chatbot.

Note: the chatbot is a local keyword/RAG-style retrieval system; it is not
an external LLM.
