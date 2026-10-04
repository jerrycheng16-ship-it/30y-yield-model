import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from fredapi import Fred

# 網頁版面設定
st.set_page_config(page_title="美國 30 年期公債殖利率總經機器學習預測 (通膨預期驅動)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率總經機器學習預測 (通膨預期驅動)</div>', unsafe_allow_html=True)
st.markdown("### 【功能說明】透過聖路易聯準會官方 FRED API 抓取真實總經數據（失業率、T5YIE 市場通膨預期、WEI 週經濟指數），結合跨資產動能，預測「下個月末」的 30 年期公債殖利率走勢、特徵歸因分析與方向勝率統計。")

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
        with st.spinner("正在透過 FRED API 同步官方真實總經數據（含 T5YIE 通膨預期與 WEI）及 Yahoo Finance 資產價格，請稍候..."):
            try:
                fred = Fred(api_key=fred_api_key.strip())
                
                # 抓取官方序列：失業率、T5YIE (5年期通膨平衡率)、WEI (週經濟指數)
                unrate = fred.get_series('UNRATE')
                t5yie = fred.get_series('T5YIE')
                wei = fred.get_series('WEI')
                
                unrate_df = pd.DataFrame({'Unemployment_Rate': unrate})
                t5yie_df = pd.DataFrame({'Inflation_Expectation': t5yie})
                wei_df = pd.DataFrame({'WEI': wei})
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
            
            # 對齊 5 年期通膨預期
            t5yie_monthly = t5yie_df.resample("ME").last()
            macro_df["Inflation_Expectation"] = t5yie_monthly["Inflation_Expectation"]
            
            # 對齊 WEI 週經濟指數
            wei_monthly = wei_df.resample("ME").last()
            macro_df["WEI"] = wei_monthly["WEI"]

            # 跨資產過去 12 個月動能 (%)
            macro_df["SP500_Mom12M"] = df_m["^GSPC"].pct_change(12) * 100
            macro_df["USD_Mom12M"] = df_m["DX-Y.NYB"].pct_change(12) * 100
            macro_df["Gold_Mom12M"] = df_m["GC=F"].pct_change(12) * 100

            # 當前實際殖利率與目標變數（未來一個月）
            macro_df["Current_TYX"] = macro_df["TYX"]
            macro_df["Target_Next_TYX"] = macro_df["TYX"].shift(-1)
            
            feature_cols = [
                "Unemployment_Rate", "Inflation_Expectation", "WEI", 
                "SP500_Mom12M", "USD_Mom12M", "Gold_Mom12M"
            ]

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

                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
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
                    "Current_TYX": test_row["Current_TYX"],
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

        # 3. 計算誤差與方向勝率
        valid_eval = results_df.dropna(subset=["Actual", "Predicted", "Current_TYX"]).copy()
        if not valid_eval.empty:
            mse = np.mean((valid_eval["Actual"] - valid_eval["Predicted"]) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_eval["Actual"] - valid_eval["Predicted"]))

            # 方向預測邏輯計算
            # 預期方向：Predicted vs Current_TYX (預期升或降)
            # 實際方向：Actual vs Current_TYX (實際升或降)
            valid_eval["Pred_Direction"] = np.where(valid_eval["Predicted"] > valid_eval["Current_TYX"], 1, -1)
            valid_eval["Actual_Direction"] = np.where(valid_eval["Actual"] > valid_eval["Current_TYX"], 1, -1)
            valid_eval["Is_Correct"] = valid_eval["Pred_Direction"] == valid_eval["Actual_Direction"]

            win_rate = valid_eval["Is_Correct"].mean() * 100
            total_trades = len(valid_eval)
            correct_trades = valid_eval["Is_Correct"].sum()
        else:
            rmse, mae, win_rate, total_trades, correct_trades = 0, 0, 0, 0, 0

        st.markdown('<div class="section-header">📊 模型表現與方向勝率摘要</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("方向預測勝率 (Win Rate)", f"{win_rate:.1f}%", f"{correct_trades}/{total_trades} 次正確")
        col2.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col3.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")
        if not results_df["Predicted"].dropna().empty:
            col4.metric("最新預測殖利率", f"{results_df['Predicted'].dropna().iloc[-1]:.2f}%")
        else:
            col4.metric("最新預測殖利率", "N/A (數據收集中)")

        # 1. 每月輸入參數與預測結果明細表（含勝率判斷）
        st.markdown('<div class="section-header">📅 每月輸入參數與「下個月預測」明細表（含方向勝率判斷）</div>', unsafe_allow_html=True)
        st.caption("說明：【預測方向】為預測值與當月實際值的比較；【實際方向】為下月實際值與當月實際值的比較。兩者相符即為預測正確（Hit）。")
        
        show_table_df = results_df[[
            "Target_Date", "Current_TYX", "Actual", "Predicted", 
            "Unemployment_Rate_Value", "Inflation_Expectation_Value", "WEI_Value",
            "SP500_Mom12M_Value", "USD_Mom12M_Value", "Gold_Mom12M_Value"
        ]].copy()

        # 計算表格中的單月勝率標記
        show_table_df["Pred_Dir"] = show_table_df["Predicted"] > show_table_df["Current_TYX"]
        show_table_df["Actual_Dir"] = show_table_df["Actual"] > show_table_df["Current_TYX"]
        show_table_df["方向勝率判斷"] = np.where(
            show_table_df["Actual"].isna() | show_table_df["Predicted"].isna(),
            "資料收集中 / 待揭曉",
            np.where(show_table_df["Pred_Dir"] == show_table_df["Actual_Dir"], "✅ 正確 (Hit)", "❌ 錯誤 (Miss)")
        )
        
        show_table_df["Target_Date"] = pd.to_datetime(show_table_df["Target_Date"]).dt.strftime("%Y-%m-%d")
        show_table_df.index = show_table_df.index.strftime("%Y-%m-%d")
        
        final_display_df = show_table_df[[
            "Target_Date", "Current_TYX", "Actual", "Predicted", "方向勝率判斷",
            "Unemployment_Rate_Value", "Inflation_Expectation_Value", "WEI_Value",
            "SP500_Mom12M_Value", "USD_Mom12M_Value", "Gold_Mom12M_Value"
        ]].copy()

        final_display_df.columns = [
            "預測目標月份 (下個月)", "當月基準實際利率", "下月實際利率", "預測殖利率", "方向預測結果",
            "失業率(%)", "5年期通膨預期 (%)", "WEI 週經濟指數",
            "S&P500動能(%)", "美元動能(%)", "黃金動能(%)"
        ]
        st.dataframe(final_display_df.round(2), use_container_width=True)

        # 2. 每個月哪一個參數影響程度最大
        st.markdown('<div class="section-header">🔍 每月參數影響程度分析（正向拉升 / 負向壓抑）</div>', unsafe_allow_html=True)
        impact_df = results_df.set_index(results_df["Target_Date"].dt.strftime("%Y-%m-%d"))[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        
        max_impact_col = impact_df.abs().idxmax(axis=1)
        impact_df["影響力最大主因"] = max_impact_col

        st.dataframe(impact_df.round(3), use_container_width=True)
