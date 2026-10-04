import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from fredapi import Fred

# 網頁版面設定
st.set_page_config(page_title="美國 30 年期公債殖利率總經機器學習預測 (零售銷售驅動)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率總經機器學習預測 (零售銷售驅動)</div>', unsafe_allow_html=True)
st.markdown("### 【功能說明】透過聖路易聯準會官方 FRED API 抓取真實月度總經數據（失業率、CPI YoY、零售銷售 YoY），結合跨資產動能，預測「下個月末」的 30 年期公債殖利率走勢與特徵歸因分析。")

# -------------------------------------------------------------
# 側邊欄參數與 FRED API Key 設定
# -------------------------------------------------------------
st.sidebar.header("⚙️ API 與回測參數設定")

default_fred_key = ""
try:
    if "FRED_API_KEY" in st.secrets:
        default_fred_key = st.secrets["FRED_API_KEY"]
except Exception:
    pass

fred_api_key = st.sidebar.text_input("FRED API Key (必填)", type="password", value=default_fred_key)
if not fred_api_key:
    st.sidebar.warning("⚠️ 請先輸入您的 FRED API Key 方可正確載入真實總經數據。")

train_window = st.sidebar.slider("訓練月數 (Train Window)", min_value=24, max_value=120, value=60, step=12)
target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2020-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始執行預測與特徵解析")

if "prediction_executed" not in st.session_state:
    st.session_state.prediction_executed = False

if run_btn:
    if not fred_api_key:
        st.error("❌ 請先在側邊欄輸入 FRED API Key！")
    else:
        with st.spinner("正在透過 FRED API 同步官方真實總經數據（含零售銷售 YoY）與 Yahoo Finance 資產價格，請稍候..."):
            try:
                fred = Fred(api_key=fred_api_key.strip())
                
                # 抓取官方月度序列：失業率、CPI、零售與餐飲銷售總額 (RSXFS)
                unrate = fred.get_series('UNRATE')
                cpi = fred.get_series('CPIAUCSL')
                retail = fred.get_series('RSXFS') 
                
                unrate_df = pd.DataFrame({'Unemployment_Rate': unrate})
                cpi_df = pd.DataFrame({'CPI': cpi})
                retail_df = pd.DataFrame({'Retail_Sales': retail})
            except Exception as e:
                st.error(f"❌ FRED API 連線或抓取失敗，請確認您的 API Key 是否正確。錯誤訊息: {e}")
                st.stop()

            # 1. 下載資產價格
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F"]
            fetch_start = pd.to_datetime("2010-01-01")
            fetch_end = pd.to_datetime(target_end_date) + pd.Timedelta(days=5)

            df_raw = yf.download(tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_prices = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]

            df_m = df_prices.resample("ME").last().ffill()
            
            macro_df = pd.DataFrame(index=df_m.index)
            macro_df["TYX"] = df_m["^TYX"]
            
            # 對齊失業率
            macro_df["Unemployment_Rate"] = unrate_df.resample("ME").last()
            
            # 計算精準的 CPI YoY (%)
            cpi_monthly = cpi_df.resample("ME").last()
            macro_df["CPI_YoY"] = cpi_monthly["CPI"].pct_change(12) * 100
            
            # 計算精準的零售銷售年增率 (Retail Sales YoY %)
            retail_monthly = retail_df.resample("ME").last()
            macro_df["Retail_Sales_YoY"] = retail_monthly["Retail_Sales"].pct_change(12) * 100

            # 跨資產過去 12 個月動能 (%)
            macro_df["SP500_Mom12M"] = df_m["^GSPC"].pct_change(12) * 100
            macro_df["USD_Mom12M"] = df_m["DX-Y.NYB"].pct_change(12) * 100
            macro_df["Gold_Mom12M"] = df_m["GC=F"].pct_change(12) * 100

            # 目標變數：未來一個月 30 年公債殖利率
            macro_df["Target_Next_TYX"] = macro_df["TYX"].shift(-1)
            
            feature_cols = [
                "Unemployment_Rate", "CPI_YoY", "Retail_Sales_YoY", 
                "SP500_Mom12M", "USD_Mom12M", "Gold_Mom12M"
            ]
            
            # 💡 關鍵修改：不強制 dropna，保留所有月份（讓抓不到數據的欄位顯示為空值以便排查）

            # 2. 機器學習與特徵歸因迴圈
            dates = macro_df.index.sort_values()
            detailed_records = []
            alphas_range = np.logspace(-2, 4, 10)

            start_idx = train_window
            if start_idx >= len(dates) - 1:
                start_idx = max(12, len(dates) // 2)

            for t in range(start_idx, len(dates)):
                test_date = dates[t]
                test_row = macro_df.iloc[t]
                target_forecast_date = test_date + pd.offsets.MonthEnd(1)

                # 檢查當期特徵是否有缺失
                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
                    # 若當期特徵完整，則進行訓練與預測
                    train_subset = macro_df.iloc[t - train_window : t].dropna(subset=feature_cols)
                    if len(train_subset) >= 12:
                        X_tr = train_subset[feature_cols]
                        y_tr = train_subset["Target_Next_TYX"].dropna()
                        common_idx = X_tr.index.intersection(y_tr.index)
                        
                        if len(common_idx) >= 12:
                            X_tr = X_tr.loc[common_idx]
                            y_tr = y_tr.loc[common_idx]
                            X_te = test_row[feature_cols].values.reshape(1, -1)

                            scaler = StandardScaler()
                            X_tr_scaled = scaler.fit_transform(X_tr)
                            X_te_scaled = scaler.transform(X_te)

                            model = RidgeCV(alphas=alphas_range).fit(X_tr_scaled, y_tr)
                            pred = model.predict(X_te_scaled)[0]
                            coefs = model.coef_
                            impacts = X_te_scaled[0] * coefs

                record = {
                    "Feature_Date": test_date,
                    "Target_Date": target_forecast_date,
                    "Actual": test_row["Target_Next_TYX"],
                    "Predicted": pred,
                }
                for i, col in enumerate(feature_cols):
                    record[f"{col}_Value"] = test_row[col]
                    record[f"{col}_Impact"] = impacts[i]

                detailed_records.append(record)

            results_df = pd.DataFrame(detailed_records)
            if not results_df.empty:
                results_df["Feature_Date"] = pd.to_datetime(results_df["Feature_Date"])
                results_df = results_df.set_index("Feature_Date")
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
        
        valid_chart_df = results_df.dropna(subset=["Predicted"])
        chart_data = valid_chart_df.set_index("Target_Date")[["Actual", "Predicted"]]
        chart_data.columns = ["實際 30 年公債殖利率 (^TYX)", "機器學習預測值"]
        st.line_chart(chart_data)

        valid_eval = results_df.dropna(subset=["Actual", "Predicted"])
        mse = np.mean((valid_eval["Actual"] - valid_eval["Predicted"]) ** 2) if not valid_eval.empty else 0
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(valid_eval["Actual"] - valid_eval["Predicted"])) if not valid_eval.empty else 0

        st.markdown('<div class="section-header">📊 模型表現摘要</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col2.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")
        if not results_df["Predicted"].dropna().empty:
            col3.metric("最新預測殖利率", f"{results_df['Predicted'].dropna().iloc[-1]:.2f}%")
        else:
            col3.metric("最新預測殖利率", "N/A (數據收集中)")

        # 1. 每月輸入參數與預測結果明細表（保留 9/30，未發布欄位留空）
        st.markdown('<div class="section-header">📅 每月輸入參數與「下個月預測」結果明細表（含最新未發布列）</div>', unsafe_allow_html=True)
        st.caption("說明：若 9/30 或最新月份的某些總經數據尚未被官方釋出，該欄位將會保持空白（NaN），方便您直接檢查哪些數據源有落後或收集問題。")
        
        show_input_df = results_df[[
            "Target_Date", "Actual", "Predicted", 
            "Unemployment_Rate_Value", "CPI_YoY_Value", "Retail_Sales_YoY_Value",
            "SP500_Mom12M_Value", "USD_Mom12M_Value", "Gold_Mom12M_Value"
        ]].copy()
        
        show_input_df["Target_Date"] = pd.to_datetime(show_input_df["Target_Date"]).dt.strftime("%Y-%m-%d")
        show_input_df.index = show_input_df.index.strftime("%Y-%m-%d")
        
        show_input_df.columns = [
            "預測目標月份 (下個月)", "實際殖利率", "預測殖利率 (針對下個月)", 
            "失業率(%)", "CPI YoY(%)", "零售銷售 YoY(%)",
            "S&P500動能(%)", "美元動能(%)", "黃金動能(%)"
        ]
        st.dataframe(show_input_df.round(2), use_container_width=True)

        # 2. 每個月哪一個參數影響程度最大
        st.markdown('<div class="section-header">🔍 每月參數影響程度分析（正向拉升 / 負向壓抑）</div>', unsafe_allow_html=True)
        impact_df = results_df.set_index(results_df["Target_Date"].dt.strftime("%Y-%m-%d"))[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        
        max_impact_col = impact_df.abs().idxmax(axis=1)
        impact_df["影響力最大主因"] = max_impact_col

        st.dataframe(impact_df.round(3), use_container_width=True)
