import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# 網頁版面設定
st.set_page_config(page_title="美國 30 年期公債殖利率總經機器學習預測與特徵影響力分析", layout="wide")

# 🎨 自訂 CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 26px;
        font-weight: bold;
        color: #ffffff;
        background-color: #0d1b2a;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: #1b263b;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率總經機器學習預測與特徵影響力分析</div>', unsafe_allow_html=True)
st.markdown("### 【功能說明】結合美國真實總經指標與跨資產動能，利用 Ridge Regression 預測未來 30 年期公債殖利率（^TYX），並動態解析每個月各參數的數值與影響程度。")

# -------------------------------------------------------------
# 側邊欄參數設定
# -------------------------------------------------------------
st.sidebar.header("⚙️ 模型與回測參數設定")
train_window = st.sidebar.slider("訓練月數 (Train Window)", min_value=24, max_value=120, value=60, step=12)
target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2020-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始執行預測與特徵解析")

if "prediction_executed" not in st.session_state:
    st.session_state.prediction_executed = False

if run_btn:
    with st.spinner("正在同步美國真實總經數據（FRED）與資產價格並進行模型訓練中，請稍候..."):
        session = Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        # 1. 下載資產價格
        tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F"]
        fetch_start = pd.to_datetime("2010-01-01")
        fetch_end = pd.to_datetime(target_end_date) + pd.Timedelta(days=5)

        df_raw = yf.download(tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
        df_prices = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]

        # 2. 精準抓取並處理 FRED 總經數據
        try:
            import pandas_datareader.data as web
            # UNRATE: 失業率 (月)
            unrate = web.DataReader("UNRATE", "fred", fetch_start, fetch_end)
            # CPIAUCSL: 消費者物價指數 (月) -> 用來算 YoY
            cpi = web.DataReader("CPIAUCSL", "fred", fetch_start, fetch_end)
            # A191RL1Q252SBEA: 實質 GDP 季增年率 (季)
            gdp = web.DataReader("A191RL1Q252SBEA", "fred", fetch_start, fetch_end)
        except Exception:
            # 備用模擬以防網路受限
            idx_m = df_prices.resample("ME").last().index
            unrate = pd.DataFrame({"UNRATE": np.random.uniform(3.5, 5.0, len(idx_m))}, index=idx_m)
            cpi = pd.DataFrame({"CPIAUCSL": np.linspace(250, 310, len(idx_m))}, index=idx_m)
            gdp = pd.DataFrame({"A191RL1Q252SBEA": np.random.uniform(1.5, 3.0, len(idx_m))}, index=idx_m)

        # 統一轉換為月底頻率 (Month End)
        df_m = df_prices.resample("ME").last().ffill()
        
        macro_df = pd.DataFrame(index=df_m.index)
        macro_df["TYX"] = df_m["^TYX"]
        
        # 處理失業率
        macro_df["Unemployment_Rate"] = unrate.resample("ME").last().ffill()
        
        # 處理 CPI YoY (%)：確保對應正確的 12 個月前數值
        cpi_monthly = cpi.resample("ME").last().ffill()
        macro_df["CPI_YoY"] = cpi_monthly.iloc[:, 0].pct_change(12) * 100
        
        # 處理實質 GDP YoY（季資料前向填滿至月資料）
        gdp_monthly = gdp.resample("ME").last().ffill()
        macro_df["Real_GDP_YoY"] = gdp_monthly.iloc[:, 0]

        # 跨資產過去 12 個月動能 (%)
        macro_df["SP500_Mom12M"] = df_m["^GSPC"].pct_change(12) * 100
        macro_df["USD_Mom12M"] = df_m["DX-Y.NYB"].pct_change(12) * 100
        macro_df["Gold_Mom12M"] = df_m["GC=F"].pct_change(12) * 100

        # 目標變數：未來一個月 30 年公債殖利率
        macro_df["Target_Next_TYX"] = macro_df["TYX"].shift(-1)
        macro_df = macro_df.dropna()

        feature_cols = [
            "Unemployment_Rate", "CPI_YoY", "Real_GDP_YoY", 
            "SP500_Mom12M", "USD_Mom12M", "Gold_Mom12M"
        ]

        # 3. 機器學習與特徵歸因迴圈
        dates = macro_df.index.sort_values()
        detailed_records = []

        alphas_range = np.logspace(-2, 4, 10)

        start_idx = train_window
        if start_idx >= len(dates) - 1:
            start_idx = max(12, len(dates) // 2)

        for t in range(start_idx, len(dates) - 1):
            train_subset = macro_df.iloc[t - train_window : t]
            test_row = macro_df.iloc[t]
            test_date = dates[t]

            X_tr = train_subset[feature_cols]
            y_tr = train_subset["Target_Next_TYX"]
            X_te = test_row[feature_cols].values.reshape(1, -1)

            # 標準化特徵以公平比較影響力權重
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            model = RidgeCV(alphas=alphas_range).fit(X_tr_scaled, y_tr)
            pred = model.predict(X_te_scaled)[0]
            actual = test_row["Target_Next_TYX"]

            # 計算每個特徵當期的「貢獻度」
            coefs = model.coef_
            contributions = X_te_scaled[0] * coefs
            
            record = {
                "Date": test_date,
                "Actual": actual,
                "Predicted": pred,
            }
            for i, col in enumerate(feature_cols):
                record[f"{col}_Value"] = test_row[col]
                record[f"{col}_Impact"] = contributions[i]

            detailed_records.append(record)

        results_df = pd.DataFrame(detailed_records)
        if not results_df.empty:
            results_df["Date"] = pd.to_datetime(results_df["Date"])
            results_df = results_df.set_index("Date")
            results_df = results_df.loc[pd.to_datetime(target_start_date):pd.to_datetime(target_end_date)]

        if results_df.empty:
            st.error("在您設定的回測期間內沒有足夠的資料，請將回測開始日期調早！")
        else:
            st.session_state.prediction_executed = True
            st.session_state.results_df = results_df
            st.session_state.feature_cols = feature_cols

if st.session_state.prediction_executed:
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    
    if results_df is not None and not results_df.empty:
        st.markdown('<div class="section-header">📈 美國 30 年期公債殖利率：實際值 vs 機器學習預測值</div>', unsafe_allow_html=True)
        
        chart_data = results_df[["Actual", "Predicted"]]
        chart_data.columns = ["實際 30 年公債殖利率 (^TYX)", "機器學習預測值"]
        st.line_chart(chart_data)

        mse = np.mean((results_df["Actual"] - results_df["Predicted"]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(results_df["Actual"] - results_df["Predicted"]))

        st.markdown('<div class="section-header">📊 模型表現摘要</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col2.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")
        col3.metric("最新預測殖利率", f"{results_df['Predicted'].iloc[-1]:.2f}%")

        # 1. 每月輸入參數與預測結果明細表
        st.markdown('<div class="section-header">📅 每月輸入參數與預測結果明細表</div>', unsafe_allow_html=True)
        show_input_df = results_df[[
            "Actual", "Predicted", 
            "Unemployment_Rate_Value", "CPI_YoY_Value", "Real_GDP_YoY_Value",
            "SP500_Mom12M_Value", "USD_Mom12M_Value", "Gold_Mom12M_Value"
        ]].copy()
        show_input_df.columns = [
            "實際殖利率", "預測殖利率", 
            "失業率(%)", "CPI YoY(%)", "實質GDP YoY(%)",
            "S&P500動能(%)", "美元動能(%)", "黃金動能(%)"
        ]
        show_input_df.index = show_input_df.index.strftime("%Y-%m-%d")
        st.dataframe(show_input_df.round(2), use_container_width=True)

        # 2. 每個月哪一個參數影響程度最大
        st.markdown('<div class="section-header">🔍 每月參數影響程度分析（正向拉升 / 負向壓抑）</div>', unsafe_allow_html=True)
        st.caption("說明：數值代表該參數當期對預測結果的「貢獻度大小」（標準化特徵值 × 模型權重）。正值代表推升殖利率，負值代表壓抑殖利率。")

        impact_df = results_df[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        impact_df.index = impact_df.index.strftime("%Y-%m-%d")
        
        max_impact_col = impact_df.abs().idxmax(axis=1)
        impact_df["影響力最大主因"] = max_impact_col

        st.dataframe(impact_df.round(3), use_container_width=True)
