import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from sklearn.linear_model import RidgeCV
from fredapi import Fred

# 網頁版面設定
st.set_page_config(page_title="美國 30 年期公債殖利率動態預測 (LightGBM + 技術指標 + MOVE波動率)", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 24px;
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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率變動量預測（LightGBM 機器學習模型）</div>', unsafe_allow_html=True)
st.markdown("### 【功能說明】結合 FRED 總經數據（失業率、T5YIE通膨預期、WEI週經濟指數）、跨資產動能、30年債技術指標（RSI、MACD乖離率）與 MOVE 美債市場波動率，預測「下個月利率變動量（ΔTYX）」。")

# -------------------------------------------------------------
# 技術指標計算輔助函數
# -------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd_diff(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line # MACD 乖離率 (Histogram)

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
        with st.spinner("正在透過 FRED API 同步總經數據、Yahoo Finance 價格、技術指標與 MOVE 美債波動率，請稍候..."):
            try:
                fred = Fred(api_key=fred_api_key.strip())
                unrate = fred.get_series('UNRATE')
                t5yie = fred.get_series('T5YIE')
                wei = fred.get_series('WEI')
                
                unrate_df = pd.DataFrame({'Unemployment_Rate': unrate})
                t5yie_df = pd.DataFrame({'Inflation_Expectation': t5yie})
                wei_df = pd.DataFrame({'WEI': wei})
            except Exception as e:
                st.error(f"❌ FRED API 連線失敗: {e}")
                st.stop()

            # 1. 下載資產價格 (包含 ^TYX, ^GSPC, DX-Y.NYB, GC=F 以及美債波動率 ^MOVE 或備援 ^VIX)
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F", "^MOVE", "^VIX"]
            fetch_start = pd.to_datetime("2010-01-01")
            fetch_end = pd.to_datetime(target_end_date) + pd.Timedelta(days=5)

            df_raw = yf.download(tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_prices = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]

            # 處理 MOVE 指數，若無則用 VIX 備援
            if "^MOVE" in df_prices.columns and df_prices["^MOVE"].notna().sum() > 50:
                move_series = df_prices["^MOVE"]
            elif "^VIX" in df_prices.columns:
                move_series = df_prices["^VIX"]
            else:
                move_series = pd.Series(20.0, index=df_prices.index)

            # 計算日頻技術指標後再 resample 至月底
            daily_df = pd.DataFrame(index=df_prices.index)
            daily_df["TYX"] = df_prices["^TYX"]
            daily_df["RSI_10"] = calculate_rsi(daily_df["TYX"], 10)
            daily_df["RSI_20"] = calculate_rsi(daily_df["TYX"], 20)
            daily_df["MACD_Diff"] = calculate_macd_diff(daily_df["TYX"])
            daily_df["Volatility_MOVE"] = move_series

            df_m = daily_df.resample("ME").last().ffill()
            prices_m = df_prices.resample("ME").last().ffill()

            macro_df = pd.DataFrame(index=df_m.index)
            macro_df["TYX"] = df_m["TYX"]
            macro_df["RSI_10"] = df_m["RSI_10"]
            macro_df["RSI_20"] = df_m["RSI_20"]
            macro_df["MACD_Diff"] = df_m["MACD_Diff"]
            macro_df["Volatility_MOVE"] = df_m["Volatility_MOVE"]
            
            # 對齊總經數據
            macro_df["Unemployment_Rate"] = unrate_df.resample("ME").last()
            t5yie_monthly = t5yie_df.resample("ME").last()
            macro_df["Inflation_Expectation"] = t5yie_monthly["Inflation_Expectation"]
            wei_monthly = wei_df.resample("ME").last()
            macro_df["WEI"] = wei_monthly["WEI"]

            # 跨資產動能 (%)
            macro_df["SP500_Mom12M"] = prices_m["^GSPC"].pct_change(12) * 100
            macro_df["USD_Mom12M"] = prices_m["DX-Y.NYB"].pct_change(12) * 100
            macro_df["Gold_Mom12M"] = prices_m["GC=F"].pct_change(12) * 100

            # 🎯 核心改動：目標變數改為「利率變動量（ΔTYX）」 = 下月實際殖利率 - 當月實際殖利率
            macro_df["Current_TYX"] = macro_df["TYX"]
            macro_df["Next_TYX"] = macro_df["TYX"].shift(-1)
            macro_df["Target_Delta_TYX"] = macro_df["Next_TYX"] - macro_df["Current_TYX"]
            
            feature_cols = [
                "Unemployment_Rate", "Inflation_Expectation", "WEI", 
                "SP500_Mom12M", "USD_Mom12M", "Gold_Mom12M",
                "RSI_10", "RSI_20", "MACD_Diff", "Volatility_MOVE"
            ]

            # 2. 機器學習迴圈（優先使用 LightGBM 樹狀模型）
            dates = macro_df.index.sort_values()
            detailed_records = []

            start_idx = train_window
            if start_idx >= len(dates) - 1:
                start_idx = max(12, len(dates) // 2)

            for t in range(start_idx, len(dates)):
                test_date = dates[t]
                test_row = macro_df.iloc[t]
                target_forecast_date = test_date + pd.offsets.MonthEnd(1)

                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred_delta = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
                    train_subset = macro_df.iloc[t - train_window : t].dropna(subset=feature_cols + ["Target_Delta_TYX"])
                    if len(train_subset) >= 12:
                        X_tr = train_subset[feature_cols]
                        y_tr = train_subset["Target_Delta_TYX"]
                        X_te = test_row[feature_cols].values.reshape(1, -1)

                        scaler = StandardScaler()
                        X_tr_scaled = scaler.fit_transform(X_tr)
                        X_te_scaled = scaler.transform(X_te)

                        if HAS_LGB:
                            model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
                            model.fit(X_tr_scaled, y_tr)
                            pred_delta = model.predict(X_te_scaled)[0]
                            # 樹狀模型特徵重要性或簡易線性近似權重
                            if hasattr(model, "feature_importances_"):
                                impacts = X_te_scaled[0] * (model.feature_importances_ / model.feature_importances_.sum())
                        else:
                            model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X_tr_scaled, y_tr)
                            pred_delta = model.predict(X_te_scaled)[0]
                            impacts = X_te_scaled[0] * model.coef_

                # 還原預測絕對利率 = 當前實際利率 + 預測變動量
                predicted_tyx = test_row["Current_TYX"] + pred_delta if not np.isnan(pred_delta) else np.nan

                record = {
                    "Feature_Date": test_date,
                    "Target_Date": target_forecast_date,
                    "Current_TYX": test_row["Current_TYX"],
                    "Actual": test_row["Next_TYX"],
                    "Predicted": predicted_tyx,
                    "Predicted_Delta": pred_delta,
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
        if not valid_chart_df.empty:
            chart_data = valid_chart_df.set_index("Target_Date")[["Actual", "Predicted"]]
            chart_data.columns = ["實際 30 年公債殖利率 (^TYX)", "機器學習預測值 (基於Δ預測)"]
            st.line_chart(chart_data)

        # 3. 計算勝率與期望值
        valid_eval = results_df.dropna(subset=["Actual", "Predicted", "Current_TYX"]).copy()
        if not valid_eval.empty:
            mse = np.mean((valid_eval["Actual"] - valid_eval["Predicted"]) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_eval["Actual"] - valid_eval["Predicted"]))

            valid_eval["Pred_Direction"] = np.where(valid_eval["Predicted"] > valid_eval["Current_TYX"], 1, -1)
            valid_eval["Actual_Direction"] = np.where(valid_eval["Actual"] > valid_eval["Current_TYX"], 1, -1)
            valid_eval["Is_Correct"] = valid_eval["Pred_Direction"] == valid_eval["Actual_Direction"]

            valid_eval["Abs_Move"] = np.abs(valid_eval["Actual"] - valid_eval["Current_TYX"])
            win_rate = valid_eval["Is_Correct"].mean()
            loss_rate = 1.0 - win_rate
            total_trades = len(valid_eval)
            correct_trades = valid_eval["Is_Correct"].sum()

            winning_moves = valid_eval.loc[valid_eval["Is_Correct"], "Abs_Move"]
            losing_moves = valid_eval.loc[~valid_eval["Is_Correct"], "Abs_Move"]
            avg_win_size = winning_moves.mean() if not winning_moves.empty else 0.0
            avg_loss_size = losing_moves.mean() if not losing_moves.empty else 0.0
            expectancy = (win_rate * avg_win_size) - (loss_rate * avg_loss_size)
        else:
            rmse, mae, win_rate, total_trades, correct_trades, expectancy = 0, 0, 0, 0, 0, 0

        st.markdown('<div class="section-header">📊 模型表現、勝率與預測期望值摘要 (LightGBM 驅動)</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("方向預測勝率 (Win Rate)", f"{win_rate * 100:.1f}%", f"{correct_trades}/{total_trades} 次正確")
        col2.metric("預測期望值 (Expectancy)", f"{expectancy:+.3f}%", "每次預測淨期望報酬")
        col3.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col4.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")

        if not results_df["Predicted"].dropna().empty:
            st.info(f"💡 **最新預測殖利率**：`{results_df['Predicted'].dropna().iloc[-1]:.2f}%`（針對下個月目標，模型架構：{'LightGBM' if HAS_LGB else 'RidgeCV'}）")

        # 1. 每月明細表
        st.markdown('<div class="section-header">📅 每月輸入參數與「下個月預測」明細表（含技術指標與波動率）</div>', unsafe_allow_html=True)
        
        show_table_df = results_df[[
            "Target_Date", "Current_TYX", "Actual", "Predicted", 
            "Unemployment_Rate_Value", "Inflation_Expectation_Value", "WEI_Value",
            "RSI_10_Value", "MACD_Diff_Value", "Volatility_MOVE_Value"
        ]].copy()

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
            "RSI_10_Value", "MACD_Diff_Value", "Volatility_MOVE_Value"
        ]].copy()

        final_display_df.columns = [
            "預測目標月份 (下個月)", "當月基準實際利率", "下月實際利率", "預測殖利率", "方向預測結果",
            "失業率(%)", "5年通膨預期(%)", "WEI週經濟", "RSI(10)", "MACD乖離", "MOVE波動率"
        ]
        st.dataframe(final_display_df.round(2), use_container_width=True)

        # 2. 影響力分析
        st.markdown('<div class="section-header">🔍 每月參數影響程度分析</div>', unsafe_allow_html=True)
        impact_df = results_df.set_index(results_df["Target_Date"].dt.strftime("%Y-%m-%d"))[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        
        if not impact_df.empty and not impact_df.isna().all().all():
            max_impact_col = impact_df.abs().idxmax(axis=1)
            impact_df["影響力最大主因"] = max_impact_col
        else:
            impact_df["影響力最大主因"] = "資料收集中"

        st.dataframe(impact_df.round(3), use_container_width=True)
