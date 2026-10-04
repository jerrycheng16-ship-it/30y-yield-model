import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from requests import Session
from sklearn.preprocessing import StandardScaler
try:import streamlit as st
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
st.set_page_config(page_title="美國 30 年期公債殖利率動態預測 (LightGBM 驅動)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率多期預測系統 (LightGBM + 總經與技術指標)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 網頁上的預測邏輯與架構說明 Tag (Expander)
# -------------------------------------------------------------
with st.expander("📖 點此展開：核心預測邏輯與模型架構說明文件", expanded=False):
    st.markdown("""
    ### 🧠 30 年期公債殖利率預測模型說明文件
    
    本系統採用非線性機器學習（**LightGBM** 樹狀模型）結合高頻總經、市場通膨預期、跨資產動能與債市技術指標，旨在解決傳統線性模型對債市轉折點反應遲鈍的問題。
    
    #### 1. 核心預測目標（Target Definition）
    * **改用變量預測（$\Delta \text{TYX}$）**：模型不直接預測殖利率的絕對數值，而是預測**未來特定跨度後的殖利率變動量**（例如：$TY_{t+k} - TY_t$）。這能有效過濾掉絕對水準的雜訊，專注於捕捉利率的升降方向。
    * **支援預測天期選擇**：
      * **1 個月期（Short-term View）**：捕捉短期月度總經發布與高頻動能衝擊。
      * **3 個月期（Medium-term View）**：透過滾動窗格預測未來一季的長天期債市趨勢，提供中長期資產配置的 Macro View。
      
    #### 2. 特徵工程矩陣（Feature Engineering）
    * **即時總經指標（無顯著發布時間差與嚴重回修）**：
      * **失業率（`UNRATE`）**：衡量勞動市場熱度。
      * **5 年期通膨預期（`T5YIE`）**：市場導向的 Breakeven Inflation Rate（每日更新，零落後）。
      * **週經濟指數（`WEI`）**：紐約聯準會高頻週度實體經濟活動指標。
    * **跨資產動能（Cross-Asset Momentum）**：
      * **S&P 500 動能（`SP500_Mom12M`）**、**美元指數動能（`USD_Mom12M`）**、**黃金動能（`Gold_Mom12M`）**。
    * **技術面與波動率特徵（Technical & Volatility）**：
      * **10 日與 20 日 RSI**、**MACD 乖離率**（捕捉債市超買超賣與動能背離）。
      * **MOVE 美債市場波動率指數（`^MOVE`，若無則以 VIX 備援）**：當美債波動率放大時，模型會動態調整特徵權重。

    #### 3. 評估與勝率統計機制（Performance Metrics）
    * **方向勝率（Directional Win Rate）**：檢驗模型預測的「升降方向」是否與實際發生一致。
    * **預測期望值（Expectancy）**：綜合勝率與實際變動幅度，計算每次預測帶來的淨期望報酬率（$\text{Win Rate} \times \text{Avg Win} - \text{Loss Rate} \times \text{Avg Loss}$）。
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
        with st.spinner(f"正在同步數據並建立「未來 {forecast_horizon} 個月」預測模型，請稍候..."):
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
            tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F", "^MOVE", "^VIX"]
            fetch_start = pd.to_datetime("2010-01-01")
            fetch_end = pd.to_datetime("2028-12-31") # 確保抓取到最前端

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
            
            macro_df["Unemployment_Rate"] = unrate_df.resample("ME").last()
            t5yie_monthly = t5yie_df.resample("ME").last()
            macro_df["Inflation_Expectation"] = t5yie_monthly["Inflation_Expectation"]
            wei_monthly = wei_df.resample("ME").last()
            macro_df["WEI"] = wei_monthly["WEI"]

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

            # 💡 允許迴圈跑到最後一個可用特徵月份（包含最新未發布未來實際值的月份）
            for t in range(start_idx, len(dates)):
                test_date = dates[t]
                test_row = macro_df.iloc[t]
                target_forecast_date = test_date + pd.offsets.MonthEnd(forecast_horizon)

                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred_delta = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
                    # 訓練集排除含有空目標值的歷史資料
                    train_subset = macro_df.iloc[:t].dropna(subset=feature_cols + ["Target_Delta_TYX"])
                    if len(train_subset) >= 12:
                        # 取最近的 train_window 筆資料進行訓練
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

if st.session_state.get("prediction_executed", False):
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    horizon_val = st.session_state.get("forecast_horizon", 1)
    
    if results_df is not None and not results_df.empty:
        st.markdown(f'<div class="section-header">📈 美國 30 年期公債殖利率：實際值 vs 預測值（未來 {horizon_val} 個月期）</div>', unsafe_allow_html=True)
        
        valid_chart_df = results_df.dropna(subset=["Predicted", "Actual"])
        if not valid_chart_df.empty:
            chart_data = valid_chart_df.set_index("Target_Date")[["Actual", "Predicted"]]
            chart_data.columns = [f"實際 30 年公債殖利率 (+{horizon_val}M)", f"機器學習預測值 (+{horizon_val}M)"]
            st.line_chart(chart_data)

        # 3. 計算歷史已實現的勝率與期望值
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

        if not results_df["Predicted"].dropna().empty:
            latest_pred_row = results_df["Predicted"].dropna().iloc[-1]
            latest_target_date = results_df["Predicted"].dropna().index[-1] + pd.DateOffset(months=horizon_val)
            st.info(f"💡 **最新即時預測**：以 `{results_df.dropna(subset=['Predicted']).index[-1].strftime('%Y-%m-%d')}` 為基準資料日，預測 **{latest_target_date.strftime('%Y-%m-%d')}** 的殖利率為 `🌿 {latest_pred_row:.2f}%`（模型：{'LightGBM' if HAS_LGB else 'RidgeCV'}）")

        # 1. 每月明細表（包含最新未揭曉的預測列）
        st.markdown(f'<div class="section-header">📅 每月輸入參數與「未來 {horizon_val} 個月預測」明細表（含最新即時預測）</div>', unsafe_allow_html=True)
        
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
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from sklearn.linear_model import RidgeCV
from fredapi import Fred

# 網頁版面設定
st.set_page_config(page_title="美國 30 年期公債殖利率進階預測系統 (LightGBM + 曲線斜率 + 信用利差)", layout="wide")

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

st.markdown('<div class="main-title">🇺🇸 美國 30 年期公債殖利率進階量化預測系統 (曲線斜率 + 信用利差 + 動能加速度)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 網頁上的預測邏輯與架構說明 Tag (Expander)
# -------------------------------------------------------------
with st.expander("📖 點此展開：核心預測邏輯與模型架構說明文件", expanded=False):
    st.markdown("""
    ### 🧠 30 年期公債殖利率進階預測模型說明文件
    
    本系統採用非線性機器學習（**LightGBM**）結合高頻總經、市場通膨預期、**殖利率曲線斜率（T10Y2Y）**、**信用利差（BAA10Y）**、多重窗口動能加速度與債市技術指標，全面升級對債市轉折點的捕捉能力。
    
    #### 1. 核心預測目標（Target Definition）
    * **變量預測（$\Delta \text{TYX}$）**：模型不直接預測殖利率的絕對數值，而是預測**未來 $k$ 個月後的殖利率變動量**（$TY_{t+k} - TY_t$）。
    * **支援預測天期選擇**：
      * **1 個月期（Short-term View）**：捕捉短期月度總經與高頻動能衝擊。
      * **3 個月期（Medium-term View）**：透過滾動窗格預測未來一季的長天期債市趨勢，提供中長期資產配置的 Macro View。
      
    #### 2. 進階特徵工程矩陣（Feature Engineering）
    * **總經與期限結構指標**：
      * **失業率（`UNRATE`）**：勞動市場熱度。
      * **5 年期通膨預期（`T5YIE`）**：市場導向 Breakeven Inflation Rate（每日更新，零落後）。
      * **週經濟指數（`WEI`）**：紐約聯準會高頻週度實體經濟活動指標。
      * **10年-2年公債利差（`T10Y2Y`）**：殖利率曲線斜率，長債方向最強領先指標。
      * **BAA公司債信用利差（`BAA10Y`）**：融資壓力與避險情緒（Flight to Quality）指標。
    * **多重窗口動能與加速度**：
      * S&P 500、美元指數、黃金之 **3 個月與 12 個月動能** 及 **動能變化率（加速度）**。
    * **技術面與波動率**：
      * 10日/20日 RSI、MACD 乖離率、**MOVE 美債市場波動率指數（`^MOVE`）**。

    #### 3. 評估與勝率統計機制（Directional Accuracy & Expectancy）
    * **方向正確性（Hit / Miss）定義**：
      * **基準點（$t$）**：當期資料日的實際殖利率（`Current_TYX`）。
      * **預測方向**：若預測殖利率 > `Current_TYX`，代表預期未來 $k$ 個月後利率會**走高（上升）**；反之預期**走低（下降）**。
      * **實際方向**：以未來 $k$ 個月後的真實結算實際利率（$t+k$）與當期 `Current_TYX` 相比。
      * **判定規則**：若模型預期的升降方向與實際結算方向相符，即判定為 **✅ 正確 (Hit)**；否則為 **❌ 錯誤 (Miss)**。例如選擇「預測未來 3 個月」時，就是拿 3 個月後的實際結算價來檢驗這一季的中長線趨勢判斷是否正確。
    * **預測期望值（Expectancy）**：綜合勝率與實際變動幅度，計算每次預測帶來的淨期望報酬率（$\text{Win Rate} \times \text{Avg Win} - \text{Loss Rate} \times \text{Avg Loss}$）。
    """)

# -------------------------------------------------------------
# 側邊欄參數與設定（訓練月數調整為 6 ~ 60）
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

# 🎯 調整訓練月數滑桿範圍為 6 ~ 60，預設值 24
train_window = st.sidebar.slider("訓練月數 (Train Window)", min_value=6, max_value=60, value=24, step=6)

min_bp_filter = st.sidebar.slider("盤整雜訊過濾門檻 (bps)", min_value=0, max_value=10, value=1, step=1, help="過濾掉實際變動小於此基點的微幅震盪月份，提升勝率評估精準度。")

target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2020-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始執行進階預測與特徵解析")

if "prediction_executed" not in st.session_state:
    st.session_state.prediction_executed = False

if run_btn:
    if not fred_api_key:
        st.error("❌ 請先在側邊欄輸入 FRED API Key！")
    else:
        with st.spinner(f"正在透過 FRED API 同步進階總經序列（T10Y2Y, BAA10Y, WEI, T5YIE）與市場數據，請稍候..."):
            try:
                fred = Fred(api_key=fred_api_key.strip())
                unrate = fred.get_series('UNRATE')
                t5yie = fred.get_series('T5YIE')
                wei = fred.get_series('WEI')
                t10y2y = fred.get_series('T10Y2Y')
                baa10y = fred.get_series('BAA10Y')
                
                macro_raw = pd.DataFrame({
                    'Unemployment_Rate': unrate,
                    'Inflation_Expectation': t5yie,
                    'WEI': wei,
                    'Term_Spread_10Y2Y': t10y2y,
                    'Credit_Spread_BAA': baa10y
                })
            except Exception as e:
                st.error(f"❌ FRED API 連線失敗: {e}")
                st.stop()

            # 1. 下載資產價格
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            tickers = ["^TYX", "^GSPC", "DX-Y.NYB", "GC=F", "^MOVE", "^VIX"]
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
            
            for col in macro_raw.columns:
                macro_df[col] = macro_raw[col].resample("ME").last()

            for asset_name, ticker in [("SP500", "^GSPC"), ("USD", "DX-Y.NYB"), ("Gold", "GC=F")]:
                p = prices_m[ticker]
                mom_3m = p.pct_change(3) * 100
                mom_12m = p.pct_change(12) * 100
                macro_df[f"{asset_name}_Mom3M"] = mom_3m
                macro_df[f"{asset_name}_Mom12M"] = mom_12m
                macro_df[f"{asset_name}_MomAccel"] = mom_3m - mom_3m.shift(3)

            macro_df["Current_TYX"] = macro_df["TYX"]
            macro_df["Future_TYX"] = macro_df["TYX"].shift(-forecast_horizon)
            macro_df["Target_Delta_TYX"] = macro_df["Future_TYX"] - macro_df["Current_TYX"]
            
            feature_cols = [
                "Unemployment_Rate", "Inflation_Expectation", "WEI", 
                "Term_Spread_10Y2Y", "Credit_Spread_BAA",
                "SP500_Mom3M", "SP500_Mom12M", "SP500_MomAccel",
                "USD_Mom3M", "USD_Mom12M", "USD_MomAccel",
                "Gold_Mom3M", "Gold_Mom12M", "Gold_MomAccel",
                "RSI_10", "RSI_20", "MACD_Diff", "Volatility_MOVE"
            ]

            dates = macro_df.index.sort_values()
            detailed_records = []

            # 配合較短的訓練窗口，將起始點門檻動態調整（至少需 6 個月資料）
            min_train_req = min(train_window, 12)
            start_idx = train_window
            if start_idx >= len(dates) - forecast_horizon:
                start_idx = max(min_train_req, len(dates) // 2)

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
                    if len(train_subset) >= min_train_req:
                        if len(train_subset) > train_window:
                            train_subset = train_subset.iloc[-train_window:]

                        X_tr = train_subset[feature_cols]
                        y_tr = train_subset["Target_Delta_TYX"]
                        X_te = test_row[feature_cols].values.reshape(1, -1)

                        scaler = StandardScaler()
                        X_tr_scaled = scaler.fit_transform(X_tr)
                        X_te_scaled = scaler.transform(X_te)

                        if HAS_LGB:
                            model = lgb.LGBMRegressor(n_estimators=70, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
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
                st.session_state.min_bp_filter = min_bp_filter

if st.session_state.get("prediction_executed", False):
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    horizon_val = st.session_state.get("forecast_horizon", 1)
    bp_thresh = st.session_state.get("min_bp_filter", 1) / 100.0
    
    if results_df is not None and not results_df.empty:
        st.markdown(f'<div class="section-header">📈 美國 30 年期公債殖利率：實際值 vs 進階預測值（未來 {horizon_val} 個月期）</div>', unsafe_allow_html=True)
        
        valid_chart_df = results_df.dropna(subset=["Predicted", "Actual"])
        if not valid_chart_df.empty:
            chart_data = valid_chart_df.set_index("Target_Date")[["Actual", "Predicted"]]
            chart_data.columns = [f"實際 30 年公債殖利率 (+{horizon_val}M)", f"進階預測值 (+{horizon_val}M)"]
            st.line_chart(chart_data)

        valid_eval = results_df.dropna(subset=["Actual", "Predicted", "Current_TYX"]).copy()
        if not valid_eval.empty:
            mse = np.mean((valid_eval["Actual"] - valid_eval["Predicted"]) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_eval["Actual"] - valid_eval["Predicted"]))

            eval_filtered = valid_eval[np.abs(valid_eval["Actual"] - valid_eval["Current_TYX"]) >= bp_thresh].copy()

            if not eval_filtered.empty:
                eval_filtered["Pred_Direction"] = np.where(eval_filtered["Predicted"] > eval_filtered["Current_TYX"], 1, -1)
                eval_filtered["Actual_Direction"] = np.where(eval_filtered["Actual"] > eval_filtered["Current_TYX"], 1, -1)
                eval_filtered["Is_Correct"] = eval_filtered["Pred_Direction"] == eval_filtered["Actual_Direction"]

                eval_filtered["Abs_Move"] = np.abs(eval_filtered["Actual"] - eval_filtered["Current_TYX"])
                win_rate = eval_filtered["Is_Correct"].mean()
                loss_rate = 1.0 - win_rate
                total_trades = len(eval_filtered)
                correct_trades = eval_filtered["Is_Correct"].sum()

                winning_moves = eval_filtered.loc[eval_filtered["Is_Correct"], "Abs_Move"]
                losing_moves = eval_filtered.loc[~eval_filtered["Is_Correct"], "Abs_Move"]
                avg_win_size = winning_moves.mean() if not winning_moves.empty else 0.0
                avg_loss_size = losing_moves.mean() if not losing_moves.empty else 0.0
                expectancy = (win_rate * avg_win_size) - (loss_rate * avg_loss_size)
            else:
                win_rate, loss_rate, total_trades, correct_trades, expectancy = 0, 0, 0, 0, 0
        else:
            rmse, mae, win_rate, total_trades, correct_trades, expectancy = 0, 0, 0, 0, 0, 0

        st.markdown(f'<div class="section-header">📊 進階模型表現、勝率與預測期望值摘要 (未來 {horizon_val} 個月期)</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("方向預測勝率 (Win Rate)", f"{win_rate * 100:.1f}%", f"{correct_trades}/{total_trades} 次有效趨勢預測")
        col2.metric("預測期望值 (Expectancy)", f"{expectancy:+.3f}%", "每次預測淨期望報酬")
        col3.metric("平均絕對誤差 (MAE)", f"{mae:.3f}%")
        col4.metric("均方根誤差 (RMSE)", f"{rmse:.3f}%")

        if not results_df["Predicted"].dropna().empty:
            latest_pred_row = results_df["Predicted"].dropna().iloc[-1]
            latest_target_date = results_df["Predicted"].dropna().index[-1] + pd.DateOffset(months=horizon_val)
            st.info(f"💡 **最新即時預測**：以 `{results_df.dropna(subset=['Predicted']).index[-1].strftime('%Y-%m-%d')}` 為基準資料日，預測 **{latest_target_date.strftime('%Y-%m-%d')}** 的殖利率為 `🌿 {latest_pred_row:.2f}%`（模型：{'LightGBM' if HAS_LGB else 'RidgeCV'}）")

        st.markdown(f'<div class="section-header">📅 每月輸入參數與「未來 {horizon_val} 個月預測」明細表</div>', unsafe_allow_html=True)
        
        show_table_df = results_df[[
            "Target_Date", "Current_TYX", "Actual", "Predicted", 
            "Unemployment_Rate_Value", "Inflation_Expectation_Value", "WEI_Value",
            "Term_Spread_10Y2Y_Value", "Credit_Spread_BAA_Value", "Volatility_MOVE_Value"
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
            "Term_Spread_10Y2Y_Value", "Credit_Spread_BAA_Value", "Volatility_MOVE_Value"
        ]].copy()

        final_display_df.columns = [
            f"預測目標月份 (+{horizon_val}M)", "當月基準實際利率", "目標期實際利率", "預測殖利率", "方向預測結果",
            "失業率(%)", "5年通膨預期(%)", "WEI週經濟", "10Y-2Y利差", "BAA信用利差", "MOVE波動率"
        ]
        st.dataframe(final_display_df.round(2), use_container_width=True)

        st.markdown(f'<div class="section-header">🔍 每月參數影響程度分析</div>', unsafe_allow_html=True)
        impact_df = results_df.set_index(results_df["Target_Date"].dt.strftime("%Y-%m-%d"))[[f"{col}_Impact" for col in feature_cols]].copy()
        impact_df.columns = feature_cols
        
        if not impact_df.empty and not impact_df.isna().all().all():
            max_impact_col = impact_df.abs().idxmax(axis=1)
            impact_df["影響力最大主因"] = max_impact_col
        else:
            impact_df["影響力最大主因"] = "資料收集中"

        st.dataframe(impact_df.round(3), use_container_width=True)
