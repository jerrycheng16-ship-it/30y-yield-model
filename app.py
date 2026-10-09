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
st.set_page_config(page_title="美國 30 年期公債殖利率多期預測系統 (總經變動量實驗)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率多期預測系統 (總經變動量實驗版)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 網頁上的預測邏輯與架構說明 Tag (Expander)
# -------------------------------------------------------------
with st.expander("📖 點此展開：核心預測邏輯與總經變動量實驗說明文件", expanded=False):
    st.markdown("""
    ### 🧠 總經變動量（Delta）特徵實驗說明
    
    本版本支援將總經指標（失業率、通膨預期、WEI週經濟指數）從**絕對水準（Level）**切換為**月變動量（$\Delta$, Monthly Diff）**：
    * **絕對水準（Level）**：捕捉經濟數據的實質高低位（例如失業率在 3.5% 還是 6%）。
    * **月變動量（Delta）**：捕捉經濟數據的**邊際變化速度**（例如失業率是加速惡化還是改善）。在債市定價中，邊際變化往往比絕對水準更能提前引發利率轉折。
    """)

# -------------------------------------------------------------
# 側邊欄參數與預測天期設定
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

forecast_horizon = st.sidebar.selectbox("選擇預測天期 (Horizon)", options=[1, 3], format_func=lambda x: f"預測未來 {x} 個月")

strategy_mode = st.sidebar.selectbox(
    "選擇債券策略模式", 
    options=["TLT + 現金 (單向多頭)", "TLT + TBT (多空雙向切換)"],
    index=0
)

macro_feature_type = st.sidebar.selectbox(
    "總經特徵呈現方式",
    options=["絕對水準 (Level)", "月變動量 (Delta / Diff)"],
    index=1 # 預設選取變動量讓您直接測試
)

train_window = st.sidebar.slider("訓練月數 (Train Window)", min_value=6, max_value=60, value=36, step=6)
target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2020-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始執行預測與策略回測")

if "prediction_executed" not in st.session_state:
    st.session_state.prediction_executed = False

if run_btn:
    if not fred_api_key:
        st.error("❌ 請先在側邊欄輸入 FRED API Key！")
    else:
        with st.spinner(f"正在同步數據（總經特徵模式：{macro_feature_type}），並執行預測與回測..."):
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

            # 1. 下載資產價格
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F", "^MOVE", "^VIX", "TLT", "TBT"]
            fetch_start = pd.to_datetime("2010-01-01")
            fetch_end = pd.to_datetime("2028-12-31")

            df_raw = yf.download(tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_prices = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]

            if "^MOVE" in df_prices.columns and df_prices["^MOVE"].notna().sum() > 50:
                move_series = df_prices["^MOVE"]
            elif "^VIX" in df_prices.columns:
                move_series = df_prices["^VIX"]
            else:
                move_series = pd.Series(20.0, index=df_prices.index)

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
                return macd - signal_line

            daily_df = pd.DataFrame(index=df_prices.index)
            daily_df["TYX"] = df_prices["^TYX"]
            daily_df["TLT"] = df_prices["TLT"]
            daily_df["TBT"] = df_prices["TBT"]
            daily_df["RSI_10"] = calculate_rsi(daily_df["TYX"], 10)
            daily_df["RSI_20"] = calculate_rsi(daily_df["TYX"], 20)
            daily_df["MACD_Diff"] = calculate_macd_diff(daily_df["TYX"])
            daily_df["Volatility_MOVE"] = move_series

            df_m = daily_df.resample("ME").last().ffill()
            prices_m = df_prices.resample("ME").last().ffill()

            macro_df = pd.DataFrame(index=df_m.index)
            macro_df["TYX"] = df_m["TYX"]
            macro_df["TLT"] = df_m["TLT"]
            macro_df["TBT"] = df_m["TBT"]
            macro_df["RSI_10"] = df_m["RSI_10"]
            macro_df["RSI_20"] = df_m["RSI_20"]
            macro_df["MACD_Diff"] = df_m["MACD_Diff"]
            macro_df["Volatility_MOVE"] = df_m["Volatility_MOVE"]
            
            # 處理總經數據（Level vs Delta）
            raw_unrate = unrate_df.resample("ME").last()
            raw_t5yie = t5yie_df.resample("ME").last()
            raw_wei = wei_df.resample("ME").last()

            if macro_feature_type == "月變動量 (Delta / Diff)":
                macro_df["Unemployment_Rate"] = raw_unrate["Unemployment_Rate"].diff()
                macro_df["Inflation_Expectation"] = raw_t5yie["Inflation_Expectation"].diff()
                macro_df["WEI"] = raw_wei["WEI"].diff()
            else:
                macro_df["Unemployment_Rate"] = raw_unrate["Unemployment_Rate"]
                macro_df["Inflation_Expectation"] = raw_t5yie["Inflation_Expectation"]
                macro_df["WEI"] = raw_wei["WEI"]

            macro_df["SP500_Mom12M"] = prices_m["^GSPC"].pct_change(12) * 100
            macro_df["USD_Mom12M"] = prices_m["DX-Y.NYB"].pct_change(12) * 100
            macro_df["Gold_Mom12M"] = prices_m["GC=F"].pct_change(12) * 100

            macro_df["Current_TYX"] = macro_df["TYX"]
            macro_df["Future_TYX"] = macro_df["TYX"].shift(-forecast_horizon)
            macro_df["Target_Delta_TYX"] = macro_df["Future_TYX"] - macro_df["Current_TYX"]
            
            feature_cols = [
                "Unemployment_Rate", "Inflation_Expectation", "WEI", 
                "SP500_Mom12M", "USD_Mom12M", "Gold_Mom12M",
                "RSI_10", "RSI_20", "MACD_Diff", "Volatility_MOVE"
            ]

            dates = macro_df.index.sort_values()
            detailed_records = []

            start_idx = train_window
            if start_idx >= len(dates) - forecast_horizon:
                start_idx = max(12, len(dates) // 2)

            for t in range(start_idx, len(dates)):
                test_date = dates[t]
                test_row = macro_df.iloc[t]
                target_forecast_date = test_date + pd.offsets.MonthEnd(forecast_horizon)

                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred_delta = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
                    train_subset = macro_df.iloc[:t].dropna(subset=feature_cols + ["Target_Delta_TYX"])
                    if len(train_subset) >= 12:
                        if len(train_subset) > train_window:
                            train_subset = train_subset.iloc[-train_window:]

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
                            if hasattr(model, "feature_importances_"):
                                impacts = X_te_scaled[0] * (model.feature_importances_ / model.feature_importances_.sum())
                        else:
                            model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X_tr_scaled, y_tr)
                            pred_delta = model.predict(X_te_scaled)[0]
                            impacts = X_te_scaled[0] * model.coef_

                predicted_tyx = test_row["Current_TYX"] + pred_delta if not np.isnan(pred_delta) else np.nan

                record = {
                    "Feature_Date": test_date,
                    "Target_Date": target_forecast_date,
                    "Current_TYX": test_row["Current_TYX"],
                    "TLT_Price": test_row["TLT"],
                    "TBT_Price": test_row["TBT"],
                    "Actual": test_row["Future_TYX"],
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
                results_df = results_df.loc[pd.to_datetime(target_start_date):pd.to_datetime(target_end_date) + pd.DateOffset(months=forecast_horizon)]

            if results_df.empty:
                st.error("在您設定的回測期間內沒有足夠的資料，請將回測開始日期調早！")
            else:
                st.session_state.prediction_executed = True
                st.session_state.results_df = results_df
                st.session_state.feature_cols = feature_cols
                st.session_state.forecast_horizon = forecast_horizon
                st.session_state.strategy_mode = strategy_mode
                st.session_state.macro_feature_type = macro_feature_type

if st.session_state.get("prediction_executed", False):
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    horizon_val = st.session_state.get("forecast_horizon", 1)
    mode_val = st.session_state.get("strategy_mode", "TLT + 現金 (單向多頭)")
    feat_type_val = st.session_state.get("macro_feature_type", "月變動量 (Delta / Diff)")
    
    if results_df is not None and not results_df.empty:
        st.markdown(f'<div class="section-header">📈 美國 30 年期公債殖利率：實際值 vs 預測值（總經模式：{feat_type_val}）</div>', unsafe_allow_html=True)
        
        valid_pred_rows = results_df.dropna(subset=["Predicted"])
        if not valid_pred_rows.empty:
            latest_row = valid_pred_rows.iloc[-1]
            latest_feature_date = valid_pred_rows.index[-1].strftime('%Y-%m-%d')
            latest_target_date = (valid_pred_rows.index[-1] + pd.DateOffset(months=horizon_val)).strftime('%Y-%m-%d')
            
            curr_rate = latest_row["Current_TYX"]
            pred_rate = latest_row["Predicted"]
            diff_rate = pred_rate - curr_rate

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("最近基準實際利率", f"{curr_rate:.2f}%", f"資料日: {latest_feature_date}")
            mcol2.metric(f"最新預測利率 (+{horizon_val}M)", f"{pred_rate:.2f}%", f"{'+' if diff_rate >= 0 else ''}{diff_rate:.2f}% vs 當前")
            mcol3.metric("預測目標結算日", latest_target_date, f"模型: {'LightGBM' if HAS_LGB else 'RidgeCV'}")

        valid_chart_df = results_df.dropna(subset=["Predicted", "Actual"])
        if not valid_chart_df.empty:
            chart_data = valid_chart_df.set_index("Target_Date")[["Actual", "Predicted"]]
            chart_data.columns = [f"實際 30 年公債殖利率 (+{horizon_val}M)", f"機器學習預測值 (+{horizon_val}M)"]
            st.line_chart(chart_data)

        # 3. 計算歷史勝率與期望值
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

        st.markdown(f'<div class="section-header">📊 模型表現、勝率與預測期望值摘要 (未來 {horizon_val} 個月期)</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("方向預測勝率 (Win Rate)", f"{win_rate * 100:.1f}%", f"{correct_trades}/{total_trades} 次正確")
        col2.metric("預測期望值 (Expectancy)", f"{expectancy:+.3f}%", "每次預測淨期望報酬")
        col3.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col4.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")

        # -------------------------------------------------------------
        # 🚀 債券策略回測引擎與績效呈現
        # -------------------------------------------------------------
        st.markdown(f'<div class="section-header">💰 債券策略回測淨值曲線與績效表現 [{mode_val} | 總經模式：{feat_type_val}]</div>', unsafe_allow_html=True)
        
        backtest_df = results_df.dropna(subset=["Predicted", "TLT_Price", "TBT_Price", "Current_TYX"]).copy()
        if not backtest_df.empty:
            tlt_ret = backtest_df["TLT_Price"].pct_change()
            tbt_ret = backtest_df["TBT_Price"].pct_change()

            if mode_val == "TLT + 現金 (單向多頭)":
                backtest_df["Signal"] = np.where(backtest_df["Predicted"] < backtest_df["Current_TYX"], 1, 0)
                backtest_df["Strategy_Return"] = backtest_df["Signal"].shift(1) * tlt_ret
                benchmark_ret = tlt_ret
            else:
                backtest_df["Signal"] = np.where(backtest_df["Predicted"] < backtest_df["Current_TYX"], 1, -1)
                strategy_ret = np.where(backtest_df["Signal"].shift(1) == 1, tlt_ret, tbt_ret)
                backtest_df["Strategy_Return"] = strategy_ret
                benchmark_ret = tlt_ret

            backtest_df["Strategy_Return"] = backtest_df["Strategy_Return"].fillna(0)
            backtest_df["Benchmark_Nav"] = (1.0 + benchmark_ret.fillna(0)).cumprod()
            backtest_df["Strategy_Nav"] = (1.0 + backtest_df["Strategy_Return"]).cumprod()

            total_days = (backtest_df.index[-1] - backtest_df.index[0]).days
            years = max(total_days / 365.25, 0.5)
            
            strat_total_return = backtest_df["Strategy_Nav"].iloc[-1] - 1.0
            strat_cagr = (backtest_df["Strategy_Nav"].iloc[-1] ** (1 / years)) - 1.0
            
            bench_total_return = backtest_df["Benchmark_Nav"].iloc[-1] - 1.0
            bench_cagr = (backtest_df["Benchmark_Nav"].iloc[-1] ** (1 / years)) - 1.0

            strat_rolling_max = backtest_df["Strategy_Nav"].cummax()
            strat_drawdown = (backtest_df["Strategy_Nav"] - strat_rolling_max) / strat_rolling_max
            strat_mdd = strat_drawdown.min()

            bench_rolling_max = backtest_df["Benchmark_Nav"].cummax()
            bench_drawdown = (backtest_df["Benchmark_Nav"] - bench_rolling_max) / bench_rolling_max
            bench_mdd = bench_drawdown.min()

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("策略年化報酬率 (CAGR)", f"{strat_cagr * 100:.2f}%", f"基准(TLT): {bench_cagr * 100:.2f}%")
            pcol2.metric("策略總報酬率", f"{strat_total_return * 100:.2f}%", f"基准: {bench_total_return * 100:.2f}%")
            pcol3.metric("策略最大回落 (MDD)", f"{strat_mdd * 100:.2f}%", f"基准: {bench_mdd * 100:.2f}%")
            pcol4.metric("回測期間", f"{years:.1f} 年", f"{len(backtest_df)} 個交易點")

            nav_chart_df = backtest_df[["Strategy_Nav", "Benchmark_Nav"]].copy()
            nav_chart_df.columns = [f"策略淨值曲線 ({mode_val})", "TLT 買入持有 (Benchmark)"]
            st.line_chart(nav_chart_df)
        else:
            st.warning("⚠️ 目前回測期間資料不足，無法計算策略績效。")

        # 1. 每月明細表
        st.markdown(f'<div class="section-header">📅 每月輸入參數與「未來 {horizon_val} 個月預測」明細表</div>', unsafe_allow_html=True)
        
        show_table_df = results_df[[
            "Target_Date", "Current_TYX", "Actual", "Predicted", 
            "Unemployment_Rate_Value", "Inflation_Expectation_Value", "WEI_Value",
            "RSI_10_Value", "MACD_Diff_Value", "Volatility_MOVE_Value"
        ]].copy()

        show_table_df["Pred_Dir"] = show_table_df["Predicted"] > show_table_df["Current_TYX"]
        show_table_df["Actual_Dir"] = show_table_df["Actual"] > show_table_df["Current_TYX"]
        show_table_df["方向勝率判斷"] = np.where(
            show_table_df["Actual"].isna() | show_table_df["Predicted"].isna(),
            "⏳ 最新即時預測 (待揭曉)",
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
            f"預測目標月份 (+{horizon_val}M)", "當月基準實際利率", "目標期實際利率", "預測殖利率", "方向預測結果",
            "失業率變動(%)" if feat_type_val.startswith("月變動量") else "失業率(%)", 
            "5年通膨預期變動(%)" if feat_type_val.startswith("月變動量") else "5年通膨預期(%)", 
            "WEI週經濟變動" if feat_type_val.startswith("月變動量") else "WEI週經濟", 
            "RSI(10)", "MACD乖離", "MOVE波動率"
        ]
        st.dataframe(final_display_df.round(3), use_container_width=True)

        # 2. 影響力分析
        st.markdown(f'<div class="section-header">🔍 每月參數影響程度分析</div>', unsafe_allow_html=True)
        impact_df = results_df.set_index(results_df["Target_Date"].dt.strftime("%Y-%m-%d"))[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        
        if not impact_df.empty and not impact_df.isna().all().all():
            try:
                max_vals = impact_df.abs().max(axis=1)
                valid_rows = max_vals > 0
                max_impact_col = pd.Series("資料收集中", index=impact_df.index)
                if valid_rows.any():
                    max_impact_col.loc[valid_rows] = impact_df.loc[valid_rows].abs().idxmax(axis=1)
                impact_df["影響力最大主因"] = max_impact_col
            except Exception:
                impact_df["影響力最大主因"] = "資料收集中"
        else:
            impact_df["影響力最大主因"] = "資料收集中"

        st.dataframe(impact_df.round(3), use_container_width=True)
