import os
import json
import traceback
from datetime import timedelta

import pandas as pd
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ==============================================================================
# 1. 初始化與設定 (Initialization & Setup)
# ==============================================================================

# 從 .env 檔案載入環境變數 (例如 GOOGLE_API_KEY)
load_dotenv()

# 初始化 FastAPI 應用
app = FastAPI(
    title="自然語言股票策略回測 API",
    description="一個能將自然語言交易策略轉換為程式碼並執行回測的 API。",
    version="11.0.0", # The Final Version
)

# 設定 CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允許所有來源的請求
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定 Google Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("環境變數 GOOGLE_API_KEY 未設定，請在 .env 檔案中加入。")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Gemini API 初始化失敗: {e}")
    # raise

# ==============================================================================
# 2. Pydantic 模型 (Data Models)
# ==============================================================================

class StrategyRequest(BaseModel):
    strategy: str

class BacktestResponse(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    stats: dict
    plot_html_url: str | None = None

class QuestionResponse(BaseModel):
    question: str

# ==============================================================================
# 3. 服務函式 (Service Functions)
# ==============================================================================

def get_llm_prompt(user_strategy: str) -> str:
    """為語言模型建立一個結構化的提示，使其能將自然語言策略轉換為 JSON"""
    return f'''
You are an expert financial analyst and Python programmer. Your task is to interpret a user's trading strategy and convert it into a structured JSON object. You must support a new, highly advanced and flexible strategy type: "COMPLEX_EVENT_TRADE".

**CRITICAL RULE:** If the strategy is ambiguous or uses features beyond what is described, you MUST return a JSON object with a "question" key asking for clarification.

**--- *** TICKER SYMBOL RULE *** ---**
**Ticker Symbol Rule:** You MUST use the '^' prefix for indices. For example, if the user mentions 'VIX', you must convert it to '^VIX'. If they mention 'Nasdaq 100', convert it to '^NDX'. For regular stocks like 'TSMC' or 'QQQ', do not add a prefix.

**--- STRATEGY TYPE: COMPLEX_EVENT_TRADE ---**
*Description*: A universal strategy engine that can handle various triggers and actions.
*Parameters*:
- `trade_asset`: The asset to be traded (e.g., "QQQ").
- `indicator_asset`: The asset used for signals (e.g., "^VIX", "^NDX").
- `buy_trigger`: The condition to initiate the first buy.
  - `type`: Can be "indicator_cross_below_ma", "indicator_cross_above_ma", "indicator_above_level", "indicator_below_level".
  - `params`: Contains `ma_period` or `level`.
- `pyramid_rules`: A list of rules for adding to the position.
  - `trigger_type`: Can be "price_drop_pct" (from last buy) or "indicator_increase_pct" (from last buy).
  - `params`: Contains `pct` and `add_size_pct` (of current equity).
- `sell_trigger`: The condition to sell the entire position.
  - `type`: Can be "take_profit_pct" or "indicator_cross_below_ma".
  - `params`: Contains `pct` or `ma_period`.

**Example 1: The user's VIX Spike Strategy**
*User's Strategy*: "Using TQQQ and VIX from 2019 to 2024. Buy 50% TQQQ when VIX price breaks above 30. Then, for every 20% increase in VIX from the last purchase, add 20% of my current equity. Do this 5 times. Sell all if my total profit exceeds 500%."
*Your JSON*:
```json
{{
    "parameters": {{ "tickers": ["TQQQ", "^VIX"], "start_date": "2019-01-01", "end_date": "2024-12-31" }},
    "strategy_definition": {{
        "type": "COMPLEX_EVENT_TRADE",
        "params": {{
            "trade_asset": "TQQQ",
            "indicator_asset": "^VIX",
            "buy_trigger": {{ "type": "indicator_above_level", "params": {{ "level": 30 }} }},
            "initial_size_pct": 50,
            "pyramid_rules": [
                {{ "trigger_type": "indicator_increase_pct", "params": {{ "pct": 20, "add_size_pct": 20 }} }},
                {{ "trigger_type": "indicator_increase_pct", "params": {{ "pct": 20, "add_size_pct": 20 }} }},
                {{ "trigger_type": "indicator_increase_pct", "params": {{ "pct": 20, "add_size_pct": 20 }} }},
                {{ "trigger_type": "indicator_increase_pct", "params": {{ "pct": 20, "add_size_pct": 20 }} }},
                {{ "trigger_type": "indicator_increase_pct", "params": {{ "pct": 20, "add_size_pct": 20 }} }}
            ],
            "sell_trigger": {{ "type": "take_profit_pct", "params": {{ "pct": 500 }} }}
        }}
    }}
}}
```

**Example 2: The user's MA Crossover Strategy**
*User's Strategy*: "Use ^NDX to trade QQQ from 1999. Buy 50% QQQ when ^NDX falls below its 365-day MA. Then for every 10% drop in QQQ price, add 10% of my equity. Do this 5 times. Sell when total profit is over 1000%."
*Your JSON*:
```json
{{
    "parameters": {{ "tickers": ["QQQ", "^NDX"], "start_date": "1999-02-11", "end_date": "2024-12-31" }},
    "strategy_definition": {{
        "type": "COMPLEX_EVENT_TRADE",
        "params": {{
            "trade_asset": "QQQ",
            "indicator_asset": "^NDX",
            "buy_trigger": {{ "type": "indicator_cross_below_ma", "params": {{ "ma_period": 365 }} }},
            "initial_size_pct": 50,
            "pyramid_rules": [
                {{ "trigger_type": "price_drop_pct", "params": {{ "pct": 10, "add_size_pct": 10 }} }},
                {{ "trigger_type": "price_drop_pct", "params": {{ "pct": 10, "add_size_pct": 10 }} }},
                {{ "trigger_type": "price_drop_pct", "params": {{ "pct": 10, "add_size_pct": 10 }} }},
                {{ "trigger_type": "price_drop_pct", "params": {{ "pct": 10, "add_size_pct": 10 }} }},
                {{ "trigger_type": "price_drop_pct", "params": {{ "pct": 10, "add_size_pct": 10 }} }}
            ],
            "sell_trigger": {{ "type": "take_profit_pct", "params": {{ "pct": 1000 }} }}
        }}
    }}
}}
```

**User's Strategy to process now:**
"{user_strategy}"

Now, generate the JSON object based on these rules.
'''

async def get_strategy_json_from_llm(user_strategy: str) -> dict:
    """呼叫 Gemini API 並解析回傳的 JSON"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = get_llm_prompt(user_strategy)
        response = await model.generate_content_async(prompt)
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"AI 回傳了格式錯誤的 JSON: {cleaned_response}")
    except Exception as e:
        print(f"呼叫 LLM 時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"與 AI 模型溝通時發生錯誤: {str(e)}")

def fetch_and_prepare_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """從 yfinance 下載資料並進行預處理"""
    print(f"正在下載 {tickers} 從 {start_date} 到 {end_date} 的資料...")
    try:
        all_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if all_data.empty or ('Close' in all_data and all_data['Close'].isnull().all().all()):
             raise ValueError(f"找不到股票代碼 {tickers} 在指定日期範圍內的有效資料。")

        primary_ticker = tickers[0]
        data = pd.DataFrame(index=all_data.index)
        
        data['Open'] = all_data['Open'][primary_ticker]
        data['High'] = all_data['High'][primary_ticker]
        data['Low'] = all_data['Low'][primary_ticker]
        data['Close'] = all_data['Close'][primary_ticker]
        data['Volume'] = all_data['Volume'][primary_ticker]

        for ticker in tickers[1:]:
            clean_ticker_name = ticker.replace("^", "").replace("=F", "")
            data[f'Close_{clean_ticker_name}'] = all_data['Close'][ticker]
            
        data = data.dropna()
        if data.empty:
            raise ValueError("資料在移除缺失值 (NaN) 後變為空，請檢查資料完整性或更換日期範圍。")
            
        print("資料下載與處理完成。")
        return data
    except Exception as e:
        print(f"資料處理時發生錯誤: {e}")
        raise HTTPException(status_code=404, detail=f"資料處理失敗: {str(e)}")

def generate_strategy_code(strategy_definition: dict) -> str:
    """根據結構化的策略定義，動態產生 backtesting.py 的 Strategy 類別程式碼"""
    params = strategy_definition.get('params', {})
    
    init_code_lines = [
        "        # --- 初始化所有策略所需的變數 ---",
        "        self.last_buy_price = 0",
        "        self.last_buy_indicator_level = 0",
        "        self.pyramid_count = 0",
        "        self.unrealized_pl = 0.0",
    ]
    
    # 動態加入指標計算
    buy_trigger = params.get('buy_trigger', {})
    if 'ma_period' in buy_trigger.get('params', {}):
        ma_period = buy_trigger['params']['ma_period']
        indicator_ticker_clean = params['indicator_asset'].replace('^', '').replace('=F', '')
        indicator_data_series = f"self.data.Close_{indicator_ticker_clean}"
        sma_name = f"self.sma_{indicator_ticker_clean}"
        init_code_lines.append(f"        {sma_name} = self.I(SMA, {indicator_data_series}, {ma_period})")

    buy_logic_lines = []
    pyramid_logic = []
    sell_logic_lines = []

    # --- 買入邏輯產生 ---
    if buy_trigger:
        initial_size_pct = params.get('initial_size_pct', 50)
        indicator_asset_clean = params['indicator_asset'].replace('^', '').replace('=F', '')
        indicator_series = f"self.data.Close_{indicator_asset_clean}"
        
        condition = "False"
        if buy_trigger['type'] == 'indicator_cross_below_ma':
            sma_name = f"self.sma_{indicator_asset_clean}"
            condition = f"crossover({sma_name}, {indicator_series})"
        elif buy_trigger['type'] == 'indicator_above_level':
            level = buy_trigger['params']['level']
            condition = f"{indicator_series}[-1] > {level}"
        
        buy_logic_lines.extend([
            f"if {condition}:",
            f"    trade_value = self.equity * {initial_size_pct} / 100",
            f"    size = int(trade_value / self.data.Close[-1])",
            f"    if size > 0:",
            f"        self.buy(size=size)",
            f"        self.last_buy_price = self.data.Close[-1]",
            f"        self.last_buy_indicator_level = {indicator_series}[-1]",
        ])

    # --- 加碼邏輯產生 ---
    pyramid_rules = params.get('pyramid_rules', [])
    if pyramid_rules:
        indicator_asset_clean = params['indicator_asset'].replace('^', '').replace('=F', '')
        indicator_series = f"self.data.Close_{indicator_asset_clean}"
        pyramid_logic.extend([
            f"if self.pyramid_count < {len(pyramid_rules)}:",
            f"    pyramid_rule = {json.dumps(pyramid_rules)}[self.pyramid_count]",
            f"    trigger_type = pyramid_rule['trigger_type']",
            f"    params = pyramid_rule['params']",
            f"    should_pyramid = False",
            f"    if trigger_type == 'price_drop_pct':",
            f"        if self.data.Close[-1] < self.last_buy_price * (1 - params['pct'] / 100):",
            f"            should_pyramid = True",
            f"    elif trigger_type == 'indicator_increase_pct':",
            f"        if {indicator_series}[-1] > self.last_buy_indicator_level * (1 + params['pct'] / 100):",
            f"            should_pyramid = True",
            f"    if should_pyramid:",
            f"        add_trade_value = self.equity * params['add_size_pct'] / 100",
            f"        add_size = int(add_trade_value / self.data.Close[-1])",
            f"        if add_size > 0 and self.equity >= self.data.Close[-1] * add_size:",
            f"            self.buy(size=add_size)",
            f"            self.last_buy_price = self.data.Close[-1]",
            f"            self.last_buy_indicator_level = {indicator_series}[-1]",
            f"            self.pyramid_count += 1",
        ])

    # --- 賣出邏輯產生 ---
    sell_trigger = params.get('sell_trigger', {})
    if sell_trigger:
        if sell_trigger['type'] == 'take_profit_pct':
            pct = sell_trigger['params']['pct']
            sell_logic_lines.extend([
                f"if self.position.pl_pct * 100 > {pct}:",
                f"    self.position.close()",
                f"    self.pyramid_count = 0",
                f"    self.last_buy_price = 0",
                f"    self.last_buy_indicator_level = 0",
            ])

    # --- 組合最終的 next 方法 ---
    next_code_lines = [
        "        if not self.position:",
        *[f"            {line}" for line in buy_logic_lines if buy_logic_lines],
        "        else:",
        *[f"            {line}" for line in pyramid_logic if pyramid_logic],
        *[f"            {line}" for line in sell_logic_lines if sell_logic_lines],
        "",
        "        if self.data.index[-1] == self.data.index.values[-1] and self.position:",
        "            self.unrealized_pl = self.position.pl_pct * 100",
    ]

    init_body = "\n".join(init_code_lines)
    next_body = "\n".join(next_code_lines)

    strategy_template = f"""
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from datetime import timedelta
import pandas as pd
import json

def SMA(arr, n):
    \"\"\"Return simple moving average of `arr` of period `n`.\"\"\"
    return pd.Series(arr).rolling(n).mean()

class GeneratedStrategy(Strategy):
    def init(self):
{init_body}
    def next(self):
{next_body}
"""
    print("--- 已產生策略程式碼 ---")
    print(strategy_template)
    print("----------------------")
    return strategy_template

def run_backtest(data: pd.DataFrame, strategy_code: str) -> dict:
    """執行回測並回傳統計結果"""
    try:
        strategy_namespace = {}
        exec(strategy_code, strategy_namespace)
        StrategyClass = strategy_namespace['GeneratedStrategy']

        bt = Backtest(data, StrategyClass, cash=10_000, commission=.002)
        stats = bt.run()
        
        stats_dict = stats.to_dict()
        
        if hasattr(stats._strategy, 'unrealized_pl') and stats._strategy.unrealized_pl != 0.0:
            stats_dict['Unrealized P&L [%]'] = stats._strategy.unrealized_pl
        
        if '_trades' in stats and not stats['_trades'].empty:
            stats_dict['# Trades'] = len(stats['_trades'])

        cleaned_stats = {k: str(v) for k, v in stats_dict.items() if not k.startswith('_')}
        
        return cleaned_stats
        
    except Exception as e:
        print(f"回測執行時發生錯誤: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"回測執行失敗: {str(e)}")

# ==============================================================================
# 4. API 端點 (API Endpoints)
# ==============================================================================

@app.get("/")
def read_root():
    return {"message": "歡迎使用自然語言股票策略回測 API"}

@app.post("/api/backtest", response_model=BacktestResponse, responses={202: {"model": QuestionResponse}})
async def perform_backtest(request: StrategyRequest):
    """
    接收自然語言策略，執行完整的回測流程。
    """
    try:
        response_data = await get_strategy_json_from_llm(request.strategy)

        if 'error' in response_data:
            raise HTTPException(status_code=400, detail=f"AI 策略解析錯誤: {response_data['error']}")
        if 'question' in response_data:
            return JSONResponse(
                status_code=202,
                content={"question": response_data['question']}
            )

        # 由於我們只有一種策略定義，直接取用
        strategy_definition = response_data.get('strategy_definition', {})
        params = response_data.get('parameters', {})
        
        data = fetch_and_prepare_data(params['tickers'], params['start_date'], params['end_date'])
        strategy_code = generate_strategy_code(strategy_definition)
        stats = run_backtest(data, strategy_code)
        
        return BacktestResponse(
            tickers=params['tickers'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            stats=stats
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"發生未預期的伺服器錯誤: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"伺服器發生未預期的錯誤: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("正在以開發模式啟動伺服器，請在另一個終端機使用 'uvicorn main:app --reload' 以獲得自動重載功能。")
    uvicorn.run(app, host="127.0.0.1", port=8000)
