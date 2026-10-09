import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
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
st.set_page_config(page_title="美國 30 年期公債殖利率多期預測系統 (交易點標示)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率多期預測系統 (含互動式進出點標示圖表)</div>', unsafe_allow_html=True)

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
    index=1
)

momentum_window = st.sidebar.selectbox(
    "跨資產動能計算週期",
    options=[1, 3, 6, 12],
    index=0, # 預設 1M 配合您的截圖
    format_func=lambda x: f"{x} 個月動能 (Mom{x}M)"
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
        with st.spinner(f"正在同步數據並執行回測..."):
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
            macro_df["Volatility_MOVE"] = move_series.resample("ME").last().ffill()
            
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

            macro_df["SP500_Mom"] = prices_m["^GSPC"].pct_change(momentum_window) * 100
            macro_df["USD_Mom"] = prices_m["DX-Y.NYB"].pct_change(momentum_window) * 100
            macro_df["Gold_Mom"] = prices_m["GC=F"].pct_change(momentum_window) * 100

            macro_df["Current_TYX"] = macro_df["TYX"]
            macro_df["Future_TYX"] = macro_df["TYX"].shift(-forecast_horizon)
            macro_df["Target_Delta_TYX"] = macro_df["Future_TYX"] - macro_df["Current_TYX"]
            
            feature_cols = [
                "Unemployment_Rate", "Inflation_Expectation", "WEI", 
                "SP500_Mom", "USD_Mom", "Gold_Mom",
                "RSI_10", "RSI_20", "MACD_Diff", "Volatility_MOVE"
            ]

            dates = macro_df.index.sort_values()
            detailed_records = []

            start_idx = max(train_window, momentum_window)
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
                st.session_state.momentum_window = momentum_window

if st.session_state.get("prediction_executed", False):
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    horizon_val = st.session_state.get("forecast_horizon", 1)
    mode_val = st.session_state.get("strategy_mode", "TLT + 現金 (單向多頭)")
    mom_val = st.session_state.get("momentum_window", 1)
    
    if results_df is not None and not results_df.empty:
        st.markdown(f'<div class="section-header">📈 美國 30 年期公債殖利率：實際值 vs 預測值（含 TLT 買進/出場時點標示）</div>', unsafe_allow_html=True)
        
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

        # -------------------------------------------------------------
        # 📊 使用 Plotly 繪製帶有進出點標示的互動式圖表
        # -------------------------------------------------------------
        valid_chart_df = results_df.dropna(subset=["Predicted", "Current_TYX"]).copy()
        if not valid_chart_df.empty:
            valid_chart_df["Signal"] = np.where(valid_chart_df["Predicted"] < valid_chart_df["Current_TYX"], 1, 0)
            
            fig = go.Figure()

            # 1. 實際利率線
            fig.add_trace(go.Scatter(
                x=valid_chart_df["Target_Date"], y=valid_chart_df["Current_TYX"],
                mode='lines', name='實際 30 年公債利率 (Current TYX)',
                line=dict(color='#00d2ff', width=2.5)
            ))

            # 2. 預測利率線
            fig.add_trace(go.Scatter(
                x=valid_chart_df["Target_Date"], y=valid_chart_df["Predicted"],
                mode='lines', name=f'機器學習預測值 (+{horizon_val}M)',
                line=dict(color='#ff9900', width=2, dash='dot')
            ))

            # 3. 標示買入 TLT 時點 (Signal == 1)
            buy_df = valid_chart_df[valid_chart_df["Signal"] == 1]
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df["Target_Date"], y=buy_df["Current_TYX"],
                    mode='markers', name='🟢 買入 TLT (預期利率降)',
                    marker=dict(color='#00ff66', size=10, symbol='triangle-up')
                ))

            # 4. 標示平倉/賣出時點 (Signal == 0)
            sell_df = valid_chart_df[valid_chart_df["Signal"] == 0]
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df["Target_Date"], y=sell_df["Current_TYX"],
                    mode='markers', name='🔴 平倉/賣出 (預期利率升)',
                    marker=dict(color='#ff3333', size=10, symbol='triangle-down')
                ))

            fig.update_layout(
                title=dict(text=f"美國 30 年期公債殖利率與策略進出點標示 (預測天期: +{horizon_val}M)", font=dict(color='white')),
                xaxis=dict(title="日期", gridcolor='#222629'),
                yaxis=dict(title="殖利率 (%)", gridcolor='#222629'),
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font=dict(color='white'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

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
        st.markdown(f'<div class="section-header">💰 債券策略回測淨值曲線與績效表現 [{mode_val}]</div>', unsafe_allow_html=True)
        
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
