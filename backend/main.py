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
    version="9.2.0", # Final Bugfix Version
)

# 設定 CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
You are an expert financial analyst and Python programmer. Your task is to interpret a user's trading strategy and convert it into a structured JSON object. You must support multiple strategy types.

**CRITICAL RULE:** If the strategy is ambiguous or uses features beyond what is described, you MUST return a JSON object with a "question" key asking for clarification.

**--- STRATEGY TYPE 1: ADVANCED_PYRAMID_TRADE ---**
*Description*: A sophisticated strategy that involves an initial buy, a series of subsequent "pyramiding" buys based on price drops, and a profit-target based sell condition.
*Example*: "I want to backtest a QQQ trading strategy using the ^NDX as a signal, from 2010 to today. The plan is: when the Nasdaq 100 index price falls below its 365-day moving average, buy an initial 50% position in QQQ. After that, for every 10% drop from the last purchase price, add another 20% of my current equity. Do this until all capital is used. Hold the entire position until the total return on it exceeds 1000%, then sell everything."
*JSON*:
```json
{{
    "parameters": {{ "tickers": ["QQQ", "^NDX"], "start_date": "2010-01-01", "end_date": "2024-12-31" }},
    "strategy_definition": {{
        "buy_rules": [{{
            "type": "ADVANCED_PYRAMID_TRADE",
            "params": {{
                "trigger_type": "CONTRARIAN_INDICATOR",
                "indicator_ticker": "^NDX",
                "ma_period": 365,
                "initial_size_pct": 50,
                "pyramid_rules": [ {{ "drop_pct": 10, "add_size_pct": 20 }}, {{ "drop_pct": 10, "add_size_pct": 20 }} ]
            }}
        }}],
        "sell_rules": [{{ "type": "TAKE_PROFIT", "params": {{ "return_pct": 1000 }} }}]
    }}
}}
```

**--- STRATEGY TYPE 2: DCA_AND_HOLD (Dollar-Cost Averaging) ---**
*Description*: A strategy that invests a fixed amount of money at regular intervals, regardless of price.
*Example*: "Backtest a Dollar-Cost Averaging strategy for QQQ from 1999 to 2024. Invest $300 every 30 days and never sell."
*JSON*:
```json
{{
    "parameters": {{ "tickers": ["QQQ"], "start_date": "1999-01-01", "end_date": "2024-12-31" }},
    "strategy_definition": {{
        "buy_rules": [{{
            "type": "DCA_AND_HOLD",
            "params": {{
                "investment_interval_days": 30,
                "investment_amount_usd": 300
            }}
        }}],
        "sell_rules": []
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
        if all_data.empty:
            raise ValueError(f"找不到股票代碼 {tickers} 在指定日期範圍內的資料。")

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
    buy_rules = strategy_definition.get('buy_rules', [])
    sell_rules = strategy_definition.get('sell_rules', [])
    strategy_type = buy_rules[0]['type'] if buy_rules else None

    init_code_lines = [
        "        # --- 初始化所有策略所需的變數 ---",
        "        self.last_buy_price = 0",
        "        self.pyramid_count = 0",
        "        self.unrealized_pl = 0.0",
        "        # DCA 策略專用變數",
        "        self.last_investment_date = -1",
        "        self.investment_interval = 0",
        "        self.investment_amount = 0",
    ]
    
    # 動態加入指標計算與參數設定
    if strategy_type == 'ADVANCED_PYRAMID_TRADE':
        buy_rule_params = buy_rules[0]['params']
        if buy_rule_params.get('trigger_type') == 'CONTRARIAN_INDICATOR':
            ma_period = buy_rule_params['ma_period']
            indicator_ticker_clean = buy_rule_params['indicator_ticker'].replace('^', '').replace('=F', '')
            indicator_data_series = f"self.data.Close_{indicator_ticker_clean}"
            sma_name = f"self.sma_{indicator_ticker_clean}"
            init_code_lines.append(f"        {sma_name} = self.I(SMA, {indicator_data_series}, {ma_period})")
    elif strategy_type == 'DCA_AND_HOLD':
        buy_rule_params = buy_rules[0]['params']
        init_code_lines.append(f"        self.investment_interval = {buy_rule_params['investment_interval_days']}")
        init_code_lines.append(f"        self.investment_amount = {buy_rule_params['investment_amount_usd']}")

    # --- *** 最終修正點 *** ---
    # 根據策略類型產生不同的 next 方法邏輯
    next_code_lines = []
    if strategy_type == 'ADVANCED_PYRAMID_TRADE':
        # --- 金字塔策略邏輯 ---
        params = buy_rules[0]['params']
        pyramid_rules = params.get('pyramid_rules', [])
        indicator_ticker_clean = params['indicator_ticker'].replace('^', '').replace('=F', '')
        indicator_data_series = f"self.data.Close_{indicator_ticker_clean}"
        sma_name = f"self.sma_{indicator_ticker_clean}"
        initial_size_pct = params['initial_size_pct']
        
        buy_logic = [
            f"if crossover({sma_name}, {indicator_data_series}):",
            f"    trade_value = self.equity * {initial_size_pct} / 100",
            f"    size = int(trade_value / self.data.Close[-1])",
            f"    if size > 0:",
            f"        self.buy(size=size)",
            f"        self.last_buy_price = self.data.Close[-1]",
        ]
        pyramid_logic = [
            f"if self.pyramid_count < {len(pyramid_rules)}:",
            f"    pyramid_rule = {json.dumps(pyramid_rules)}[self.pyramid_count]",
            f"    if self.data.Close[-1] < self.last_buy_price * (1 - pyramid_rule['drop_pct'] / 100):",
            f"        add_trade_value = self.equity * pyramid_rule['add_size_pct'] / 100",
            f"        add_size = int(add_trade_value / self.data.Close[-1])",
            f"        if add_size > 0 and self.equity >= self.data.Close[-1] * add_size:",
            f"            self.buy(size=add_size)",
            f"            self.last_buy_price = self.data.Close[-1]",
            f"            self.pyramid_count += 1",
        ]
        sell_logic = []
        if sell_rules and sell_rules[0]['type'] == 'TAKE_PROFIT':
            sell_params = sell_rules[0]['params']
            return_pct = sell_params['return_pct']
            sell_logic.extend([
                f"if self.position.pl_pct * 100 > {return_pct}:",
                f"    self.position.close()",
                f"    self.pyramid_count = 0",
                f"    self.last_buy_price = 0",
            ])
        
        next_code_lines.extend([
            "        if not self.position:",
            *[f"            {line}" for line in buy_logic],
            "        else:",
            *[f"            {line}" for line in pyramid_logic],
            *[f"            {line}" for line in sell_logic],
        ])

    elif strategy_type == 'DCA_AND_HOLD':
        # --- 定期定額策略邏輯 ---
        next_code_lines.extend([
            "        # 定期定額買入邏輯，不論是否已有倉位",
            "        if self.last_investment_date == -1 or (len(self.data) - 1 - self.last_investment_date) >= self.investment_interval:",
            "            # 計算可以購買的整數股數",
            "            size = int(self.investment_amount / self.data.Close[-1])",
            "            if self.equity >= self.investment_amount and size > 0:",
            "                self.buy(size=size)",
            "                self.last_investment_date = len(self.data) - 1",
        ])

    # 在所有策略的結尾都加上未實現損益的計算
    next_code_lines.extend([
        "",
        "        # 在回測的最後一天，如果仍有持倉，則記錄下未實現損益",
        "        if self.data.index[-1] == self.data.index.values[-1] and self.position:",
        "            self.unrealized_pl = self.position.pl_pct * 100",
    ])

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

        params = response_data['parameters']
        strategy_definition = response_data['strategy_definition']

        data = fetch_and_prepare_data(params['tickers'], params['start_date'], params['end_date'])
        strategy_code = generate_strategy_code(strategy_definition)
        stats = run_backtest(data, strategy_code)
        
        return BacktestResponse(
            tickers=params['tickers'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            stats=stats
        )

    except Exception as e:
        print(f"發生未預期的伺服器錯誤: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"伺服器發生未預期的錯誤: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("正在以開發模式啟動伺服器，請在另一個終端機使用 'uvicorn main:app --reload' 以獲得自動重載功能。")
    uvicorn.run(app, host="127.0.0.1", port=8000)
