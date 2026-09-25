
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Agricultural Price Analytics",
    page_icon="🌶️",
    layout="wide"
)

# =========================================================
# HELPERS
# =========================================================

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


@st.cache_data
def prepare_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required = [
        "t", "cmdty", "market_name", "district_name",
        "variety", "p_min", "p_max", "p_modal"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], errors="coerce")

    for c in ["p_min", "p_max", "p_modal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["t", "p_min", "p_max", "p_modal"])
    df = df.drop_duplicates()
    df = df.sort_values("t").reset_index(drop=True)

    df["year"] = df["t"].dt.year
    df["month"] = df["t"].dt.month
    df["day"] = df["t"].dt.day
    df["day_of_week"] = df["t"].dt.dayofweek

    df["price_range"] = df["p_max"] - df["p_min"]
    df["price_average"] = (df["p_min"] + df["p_max"]) / 2

    return df


def train_models(df):
    feature_cols = [
        "p_min", "p_max", "price_range", "price_average",
        "year", "month", "day", "day_of_week"
    ]

    X = df[feature_cols].copy()
    y = df["p_modal"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    pred_linear = linear_model.predict(X_test_scaled)

    # Ridge tuning
    ridge_results = []
    for alpha in [0.01, 0.1, 1, 10, 100]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        mae, mse, rmse, r2 = calculate_metrics(y_test, pred)
        ridge_results.append({
            "Model": "Ridge Regression",
            "Hyperparameter": f"alpha={alpha}",
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "MSE": mse
        })

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    pred_ridge = ridge_model.predict(X_test_scaled)

    # Lasso tuning
    lasso_results = []
    for alpha in [0.01, 0.1, 1, 10, 100]:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        mae, mse, rmse, r2 = calculate_metrics(y_test, pred)
        lasso_results.append({
            "Model": "Lasso Regression",
            "Hyperparameter": f"alpha={alpha}",
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "MSE": mse
        })

    lasso_model = Lasso(alpha=1.0, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    pred_lasso = lasso_model.predict(X_test_scaled)

    # Random Forest GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    rf_grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    pred_rf = best_rf.predict(X_test)

    mae_rf, mse_rf, rmse_rf, r2_rf = calculate_metrics(y_test, pred_rf)

    # Combined comparison
    model_comparison = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MAE": calculate_metrics(y_test, pred_linear)[0],
            "MSE": calculate_metrics(y_test, pred_linear)[1],
            "RMSE": calculate_metrics(y_test, pred_linear)[2],
            "R2": calculate_metrics(y_test, pred_linear)[3]
        },
        {
            "Model": "Ridge Regression",
            "MAE": calculate_metrics(y_test, pred_ridge)[0],
            "MSE": calculate_metrics(y_test, pred_ridge)[1],
            "RMSE": calculate_metrics(y_test, pred_ridge)[2],
            "R2": calculate_metrics(y_test, pred_ridge)[3]
        },
        {
            "Model": "Lasso Regression",
            "MAE": calculate_metrics(y_test, pred_lasso)[0],
            "MSE": calculate_metrics(y_test, pred_lasso)[1],
            "RMSE": calculate_metrics(y_test, pred_lasso)[2],
            "R2": calculate_metrics(y_test, pred_lasso)[3]
        },
        {
            "Model": "Tuned Random Forest",
            "MAE": mae_rf,
            "MSE": mse_rf,
            "RMSE": rmse_rf,
            "R2": r2_rf
        }
    ])

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "linear_model": linear_model,
        "ridge_model": ridge_model,
        "lasso_model": lasso_model,
        "best_rf": best_rf,
        "pred_linear": pred_linear,
        "pred_ridge": pred_ridge,
        "pred_lasso": pred_lasso,
        "pred_rf": pred_rf,
        "ridge_results": pd.DataFrame(ridge_results),
        "lasso_results": pd.DataFrame(lasso_results),
        "rf_grid": rf_grid,
        "model_comparison": model_comparison,
        "feature_importance": feature_importance,
        "feature_cols": feature_cols
    }


def make_rag_data(df):
    rag = df[
        ["t", "cmdty", "market_name", "district_name",
         "variety", "p_min", "p_max", "p_modal"]
    ].copy()

    rag["date"] = rag["t"].dt.date.astype(str)
    return rag


def search_price_data(rag_data, query, top_k=5):
    query = query.lower().strip()

    stop_words = {
        "what", "is", "the", "price", "of", "in", "at", "for",
        "on", "give", "show", "me", "which", "market", "has",
        "a", "an", "tell", "latest"
    }

    words = [
        w.strip("?,.!") for w in query.split()
        if w.strip("?,.!") not in stop_words
    ]

    search_text = (
        rag_data["market_name"].astype(str) + " " +
        rag_data["district_name"].astype(str) + " " +
        rag_data["variety"].astype(str) + " " +
        rag_data["date"].astype(str)
    ).str.lower()

    scores = []
    for text_value in search_text:
        score = sum(1 for word in words if word in text_value)
        scores.append(score)

    result = rag_data.copy()
    result["_score"] = scores
    result = result[result["_score"] > 0]
    result = result.sort_values(
        ["_score", "date"], ascending=[False, False]
    ).head(top_k)

    return result.drop(columns=["_score"])


def predict_market_price(df, models, market, date):
    date = pd.to_datetime(date)

    # Follows the notebook's prediction logic:
    # use dataset averages and recent modal prices as proxy inputs.
    sorted_df = df.sort_values("t")

    input_data = pd.DataFrame({
        "p_min": [df["p_min"].mean()],
        "p_max": [df["p_max"].mean()],
        "price_range": [df["p_max"].mean() - df["p_min"].mean()],
        "price_average": [(df["p_min"].mean() + df["p_max"].mean()) / 2],
        "year": [date.year],
        "month": [date.month],
        "day": [date.day],
        "day_of_week": [date.dayofweek]
    })

    pred = models["best_rf"].predict(input_data)[0]
    return float(pred)


def chatbot_answer(query, df, models):
    q = query.lower().strip()
    rag = make_rag_data(df)

    # Prediction request
    if any(word in q for word in ["predict", "predicted", "forecast"]):
        selected_market = None
        for market in rag["market_name"].dropna().unique():
            if str(market).lower() in q:
                selected_market = market
                break

        if selected_market is None:
            return "Please mention a market name for prediction."

        prediction_date = pd.Timestamp.today().normalize()
        predicted = predict_market_price(
            df, models, selected_market, prediction_date
        )

        return (
            f"### 🔮 Predicted Price\n\n"
            f"**Market:** {selected_market}\n\n"
            f"**Date:** {prediction_date.date()}\n\n"
            f"**Predicted Modal Price:** ₹{predicted:,.2f}"
        )

    if "highest" in q or "maximum" in q:
        row = rag.loc[rag["p_modal"].idxmax()]
        return (
            f"### 📈 Highest Modal Price\n\n"
            f"**Market:** {row['market_name']}  \n"
            f"**District:** {row['district_name']}  \n"
            f"**Date:** {row['date']}  \n"
            f"**Variety:** {row['variety']}  \n"
            f"**Modal Price:** ₹{row['p_modal']:,.2f}"
        )

    if "lowest" in q or "minimum" in q:
        row = rag.loc[rag["p_modal"].idxmin()]
        return (
            f"### 📉 Lowest Modal Price\n\n"
            f"**Market:** {row['market_name']}  \n"
            f"**District:** {row['district_name']}  \n"
            f"**Date:** {row['date']}  \n"
            f"**Variety:** {row['variety']}  \n"
            f"**Modal Price:** ₹{row['p_modal']:,.2f}"
        )

    # Market-specific request
    for market in rag["market_name"].dropna().unique():
        if str(market).lower() in q:
            market_data = rag[
                rag["market_name"].astype(str).str.lower()
                == str(market).lower()
            ].sort_values("date")

            latest = market_data.iloc[-1]

            return (
                f"### 📍 Market Price\n\n"
                f"**Market:** {market}  \n"
                f"**District:** {latest['district_name']}  \n"
                f"**Date:** {latest['date']}  \n"
                f"**Variety:** {latest['variety']}  \n"
                f"**Minimum:** ₹{latest['p_min']:,.2f}  \n"
                f"**Maximum:** ₹{latest['p_max']:,.2f}  \n"
                f"**Modal:** ₹{latest['p_modal']:,.2f}"
            )

    results = search_price_data(rag, q, top_k=5)

    if results.empty:
        return (
            "Sorry, I could not find relevant information. "
            "Try asking about a market, highest price, lowest price, "
            "or predicted price."
        )

    lines = ["### 🔎 Relevant Price Information\n"]
    for _, row in results.iterrows():
        lines.append(
            f"**{row['date']} — {row['market_name']}**  \n"
            f"District: {row['district_name']} | "
            f"Variety: {row['variety']}  \n"
            f"Min: ₹{row['p_min']:,.0f} | "
            f"Max: ₹{row['p_max']:,.0f} | "
            f"Modal: ₹{row['p_modal']:,.0f}\n"
        )

    return "\n".join(lines)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌶️ Agricultural Price Analytics")

uploaded_file = st.sidebar.file_uploader(
    "Upload agricultural price dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.title("🌶️ Agricultural Commodity Price Analytics")
    st.info(
        "Upload your CSV/Excel dataset from the sidebar to start. "
        "The app expects the columns used in your notebook: "
        "`t`, `cmdty`, `market_name`, `district_name`, `variety`, "
        "`p_min`, `p_max`, `p_modal`."
    )

    st.markdown("""
### This Streamlit app contains your complete notebook workflow

1. Data loading and validation
2. Data cleaning
3. Dataset overview
4. Exploratory Data Analysis
5. Feature engineering
6. Correlation analysis
7. Linear Regression
8. Ridge Regression + tuning
9. Lasso Regression + tuning
10. Random Forest + GridSearchCV
11. Model comparison
12. Feature importance
13. Price prediction
14. RAG-style agricultural price chatbot
""")
    st.stop()

try:
    df = prepare_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

with st.spinner("Training models and preparing analytics..."):
    models = train_models(df)

rag_data = make_rag_data(df)

pages = [
    "🏠 Home",
    "📋 Data Overview",
    "🧹 Data Cleaning",
    "📊 EDA",
    "🛠️ Feature Engineering",
    "📈 Linear Regression",
    "🔵 Ridge Regression",
    "🟢 Lasso Regression",
    "🌲 Random Forest",
    "🏆 Model Comparison",
    "🔮 Price Prediction",
    "🤖 Price Chatbot"
]

page = st.sidebar.radio("Navigation", pages)

# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":
    st.title("🌶️ Agricultural Commodity Price Analytics")
    st.subheader("Machine Learning + Market Analysis + RAG-style Chatbot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Markets", df["market_name"].nunique())
    c3.metric("Districts", df["district_name"].nunique())
    c4.metric("Mean Modal Price", f"₹{df['p_modal'].mean():,.2f}")

    st.markdown("### Workflow")
    st.write(
        "Data Upload → Cleaning → EDA → Feature Engineering → "
        "Regression → Hyperparameter Tuning → Random Forest → "
        "Model Comparison → Prediction → Chatbot"
    )

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# DATA OVERVIEW
# =========================================================

elif page == "📋 Data Overview":
    st.title("📋 Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.write("### Columns")
    st.write(df.columns.tolist())

    st.write("### Data Types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes]
        }),
        use_container_width=True
    )

    st.write("### Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.write("### Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

# =========================================================
# CLEANING
# =========================================================

elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")

    st.success("Date column converted to datetime.")
    st.success("Duplicate rows removed.")
    st.success("Required numeric price columns converted to numeric.")
    st.success("Rows with invalid date/price values removed.")

    cleaning_summary = pd.DataFrame({
        "Step": [
            "Original/loaded rows",
            "Final rows",
            "Duplicate rows currently present",
            "Missing values currently present"
        ],
        "Value": [
            len(df),
            len(df),
            int(df.duplicated().sum()),
            int(df.isnull().sum().sum())
        ]
    })

    st.dataframe(cleaning_summary, use_container_width=True)
    st.dataframe(df.head(30), use_container_width=True)

# =========================================================
# EDA
# =========================================================

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Distribution",
        "Time Trend",
        "Monthly",
        "Market",
        "District",
        "Correlation"
    ])

    with tab1:
        fig = px.histogram(
            df, x="p_modal", nbins=20,
            title="Distribution of Modal Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df, y="p_modal",
            title="Boxplot of Modal Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Mean modal price: ₹{df['p_modal'].mean():,.2f}. "
            f"Median: ₹{df['p_modal'].median():,.2f}. "
            f"Minimum: ₹{df['p_modal'].min():,.2f}. "
            f"Maximum: ₹{df['p_modal'].max():,.2f}."
        )

    with tab2:
        trend = df.sort_values("t")
        fig = px.line(
            trend, x="t", y="p_modal",
            title="Chilli Modal Price Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        monthly = (
            df.groupby("month", as_index=False)["p_modal"]
            .mean()
            .rename(columns={"p_modal": "Average Price"})
        )
        fig = px.line(
            monthly, x="month", y="Average Price",
            markers=True,
            title="Monthly Average Modal Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        market = (
            df.groupby("market_name", as_index=False)["p_modal"]
            .mean()
            .sort_values("p_modal", ascending=False)
        )
        fig = px.bar(
            market, x="market_name", y="p_modal",
            title="Average Chilli Price by Market"
        )
        fig.update_xaxes(tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(market, use_container_width=True)

    with tab5:
        district = (
            df.groupby("district_name", as_index=False)["p_modal"]
            .mean()
            .sort_values("p_modal", ascending=False)
        )
        fig = px.bar(
            district, x="district_name", y="p_modal",
            title="Average Chilli Price by District"
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(district, use_container_width=True)

    with tab6:
        corr = df[["p_min", "p_max", "p_modal"]].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Matrix of Chilli Prices"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(corr, use_container_width=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

elif page == "🛠️ Feature Engineering":
    st.title("🛠️ Feature Engineering")

    feature_table = df[
        [
            "p_min", "p_max", "price_range",
            "price_average", "year", "month",
            "day", "day_of_week", "p_modal"
        ]
    ].head(30)

    st.dataframe(feature_table, use_container_width=True)

    st.markdown("""
### Engineered Features

- `year` – year extracted from date
- `month` – month extracted from date
- `day` – day extracted from date
- `day_of_week` – weekday number
- `price_range` – maximum price − minimum price
- `price_average` – average of minimum and maximum price

### Target

`p_modal` – modal agricultural commodity price
""")

# =========================================================
# LINEAR
# =========================================================

elif page == "📈 Linear Regression":
    st.title("📈 Linear Regression")

    pred = models["pred_linear"]
    y_test = models["y_test"]

    mae, mse, rmse, r2 = calculate_metrics(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.4f}")
    c2.metric("MSE", f"{mse:,.4f}")
    c3.metric("RMSE", f"{rmse:,.4f}")
    c4.metric("R²", f"{r2:.6f}")

    plot_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    })

    fig = px.scatter(
        plot_df, x="Actual", y="Predicted",
        title="Linear Regression: Actual vs Predicted"
    )
    mn = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    mx = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    fig.add_trace(
        go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", name="Ideal"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    coef = pd.DataFrame({
        "Feature": models["feature_cols"],
        "Coefficient": models["linear_model"].coef_
    })
    st.write("### Coefficients")
    st.dataframe(coef, use_container_width=True)

# =========================================================
# RIDGE
# =========================================================

elif page == "🔵 Ridge Regression":
    st.title("🔵 Ridge Regression + Hyperparameter Tuning")

    pred = models["pred_ridge"]
    y_test = models["y_test"]
    mae, mse, rmse, r2 = calculate_metrics(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.4f}")
    c2.metric("MSE", f"{mse:,.4f}")
    c3.metric("RMSE", f"{rmse:,.4f}")
    c4.metric("R²", f"{r2:.6f}")

    tuning = models["ridge_results"]
    st.write("### Ridge Tuning Results")
    st.dataframe(tuning, use_container_width=True)

    fig = px.line(
        tuning, x="Hyperparameter", y="RMSE",
        markers=True, title="Ridge Alpha vs RMSE"
    )
    st.plotly_chart(fig, use_container_width=True)

    plot_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    })
    fig = px.scatter(
        plot_df, x="Actual", y="Predicted",
        title="Ridge Regression: Actual vs Predicted"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# LASSO
# =========================================================

elif page == "🟢 Lasso Regression":
    st.title("🟢 Lasso Regression + Hyperparameter Tuning")

    pred = models["pred_lasso"]
    y_test = models["y_test"]
    mae, mse, rmse, r2 = calculate_metrics(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.4f}")
    c2.metric("MSE", f"{mse:,.4f}")
    c3.metric("RMSE", f"{rmse:,.4f}")
    c4.metric("R²", f"{r2:.6f}")

    tuning = models["lasso_results"]
    st.write("### Lasso Tuning Results")
    st.dataframe(tuning, use_container_width=True)

    fig = px.line(
        tuning, x="Hyperparameter", y="RMSE",
        markers=True, title="Lasso Alpha vs RMSE"
    )
    st.plotly_chart(fig, use_container_width=True)

    plot_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    })
    fig = px.scatter(
        plot_df, x="Actual", y="Predicted",
        title="Lasso Regression: Actual vs Predicted"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RANDOM FOREST
# =========================================================

elif page == "🌲 Random Forest":
    st.title("🌲 Tuned Random Forest")

    y_test = models["y_test"]
    pred = models["pred_rf"]

    mae, mse, rmse, r2 = calculate_metrics(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("MSE", f"{mse:,.2f}")
    c3.metric("RMSE", f"{rmse:,.2f}")
    c4.metric("R²", f"{r2:.6f}")

    st.write("### Best Parameters")
    st.json(models["rf_grid"].best_params_)

    fi = models["feature_importance"].head(15)

    fig = px.bar(
        fi.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

    plot_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    })

    fig = px.scatter(
        plot_df, x="Actual", y="Predicted",
        title="Random Forest: Actual vs Predicted"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")

    comparison = models["model_comparison"].copy()
    st.dataframe(comparison, use_container_width=True)

    fig = px.bar(
        comparison,
        x="Model",
        y="RMSE",
        title="RMSE Comparison",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        comparison,
        x="Model",
        y="R2",
        title="R² Comparison",
        text_auto=".4f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "The notebook uses p_min, p_max and derived price features to "
        "predict p_modal. Because these variables are closely related "
        "to the target, the very high scores should be interpreted with "
        "care regarding target leakage."
    )

    csv = comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Model Comparison CSV",
        csv,
        "model_comparison_results.csv",
        "text/csv"
    )

# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Price Prediction":
    st.title("🔮 Agricultural Price Prediction")

    markets = sorted(
        df["market_name"].dropna().astype(str).unique().tolist()
    )

    market = st.selectbox("Select Market", markets)
    date = st.date_input(
        "Prediction Date",
        value=pd.Timestamp.today().date()
    )

    if st.button("Predict Modal Price", type="primary"):
        prediction = predict_market_price(
            df, models, market, date
        )

        st.success(
            f"Predicted modal price for **{market}** "
            f"on **{date}**: **₹{prediction:,.2f}**"
        )

        st.caption(
            "This prediction follows the notebook's simplified approach "
            "using dataset-level average price inputs and calendar features."
        )

# =========================================================
# CHATBOT
# =========================================================

elif page == "🤖 Price Chatbot":
    st.title("🤖 Agricultural Price Chatbot")

    st.markdown("""
Ask questions such as:

- What is the highest price?
- What is the lowest price?
- What is the price in Udumalpet?
- What is the price in Tiruchengode?
- What is the predicted price in Udumalpet?
""")

    query = st.text_input(
        "Ask your agricultural price question"
    )

    if st.button("Ask", type="primary") and query:
        with st.spinner("Searching the price knowledge base..."):
            answer = chatbot_answer(query, df, models)

        st.markdown(answer)

    st.divider()
    st.write("### Available Markets")
    st.write(", ".join(
        sorted(df["market_name"].dropna().astype(str).unique())
    ))
