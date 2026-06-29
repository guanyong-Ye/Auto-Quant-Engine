import os
import sys
import re
import time
import json
import warnings
import socket
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🚨 【核心防死锁】设置全局 Socket 强制超时为 15 秒，彻底断绝第三方 API 无限期挂起的可能性！
socket.setdefaulttimeout(15.0)

import pandas as pd
import akshare as ak
import baostock as bs
import lightgbm as lgb
from tqdm import tqdm
from openai import OpenAI

# ================= 运行环境与编码修复 =================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ================= 核心全局配置区 =================
# 🔑 1. API 秘钥配置 (从环境变量读取，严禁硬编码！)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL_NAME = "gemini-3.5-flash"  # 使用高速、高并发的官方主流模型

# 📁 2. 动态目录配置 (自适应当前运行环境)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "data")
HIST_DATA_DIR = os.path.join(BASE_DIR, "historical_daily")   # 历史 K 线底座目录
ALT_DATA_DIR = os.path.join(BASE_DIR, "daily_features")      # 今日情报与新闻目录
LLM_OUT_DIR = os.path.join(BASE_DIR, "llm_features")         # 大模型特征提取目录
MERGED_DATA_DIR = os.path.join(BASE_DIR, "merged_dataset")   # 最终缝合数据集目录
MODEL_DIR = os.path.join(BASE_DIR, "models", "lgbm")         # 模型权重归档目录

# 💰 3. 资金交易管理配置
TOTAL_CAPITAL = 30000
HOT_CAPITAL = TOTAL_CAPITAL * 0.35               # 游资突击队预算 (35%)
STEADY_CAPITAL = TOTAL_CAPITAL * 0.65            # 稳健主力池预算 (65%)
HOT_SLOTS, STEADY_SLOTS = 3, 4                   # 计划最大挂单槽位

# ==================================================

def init_environment():
    """初始化系统所需的文件夹结构"""
    for folder in [HIST_DATA_DIR, ALT_DATA_DIR, LLM_OUT_DIR, MERGED_DATA_DIR, MODEL_DIR]:
        os.makedirs(folder, exist_ok=True)
    print(f"📁 [系统初始化] 运行目录结构自检并准备完毕: {BASE_DIR}")
    
    if not GEMINI_API_KEY:
        print("⚠️ [安全警告] 未检测到 GEMINI_API_KEY 环境变量，大模型特征提取模块可能失效！")

# ==========================================
# 🛡️ 智能网络代理自适应切换引擎
# ==========================================
class BypassProxyContext:
    """临时清除本地环境变量中的 Proxy 代理设置，确保国内财经 API 成功连接且不被劫持"""
    def __enter__(self):
        self.old_proxies = {}
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
            if key in os.environ:
                self.old_proxies[key] = os.environ[key]
                del os.environ[key]
        return self
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时恢复原有代理环境，保证海外大模型请求正常
        for key, val in self.old_proxies.items():
            if val is not None:
                os.environ[key] = val

def get_latest_trade_date():
    """智能定位最新的 A 股有效交易日"""
    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        trade_cal['trade_date'] = pd.to_datetime(trade_cal['trade_date']).dt.date
        today = datetime.now().date()
        past_trade_days = trade_cal[trade_cal['trade_date'] <= today]['trade_date'].tolist()
        latest_trade_date = past_trade_days[-1]
        
        current_time = datetime.now()
        if latest_trade_date == today and current_time.hour < 17:
            latest_trade_date = past_trade_days[-2]
        return latest_trade_date.strftime("%Y%m%d"), latest_trade_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"⚠️ 获取交易日历失败，使用当前系统时间兜底: {e}")
        return datetime.now().strftime("%Y%m%d"), datetime.now().strftime("%Y-%m-%d")

def check_macro_risk(latest_date):
    """拉取上证指数，判断系统性宏观风险"""
    try:
        sh_index = ak.stock_zh_index_daily(symbol="sh000001")
        today_data = sh_index[sh_index['date'] == latest_date]
        if not today_data.empty:
            pct_chg = today_data['pct_chg'].iloc[0] 
            if pct_chg <= -2.0:
                return True 
    except Exception:
        pass
    return False

# ==========================================
# 阶段一：BaoStock 极速缓存命中与采集引擎
# ==========================================
def update_daily_price_with_baostock(TODAY_DATE, TODAY_STR):
    print(f"\n🔄 [阶段一] 启动 BaoStock 行情引擎，拉取今日 ({TODAY_DATE}) 全市场行情...")
    files = [f for f in os.listdir(HIST_DATA_DIR) if f.endswith('.csv')]
    if len(files) == 0:
        print("⚠️ 历史数据目录为空，请先填充底座历史K线数据。")
        return None

    stock_name_cache = {}
    def load_name(file):
        code = file.split('.')[0].split('_')[0]
        try:
            df_meta = pd.read_csv(os.path.join(HIST_DATA_DIR, file), nrows=1)
            if not df_meta.empty and '股票名称' in df_meta.columns:
                return code, df_meta['股票名称'].iloc[0]
        except: pass
        return code, None

    with ThreadPoolExecutor(max_workers=32) as executor:
        for code, name in executor.map(load_name, files):
            if name: stock_name_cache[code] = name

    bs.login()
    daily_snapshot_list, update_count, cache_hit_count = [], 0, 0

    pbar = tqdm(files, desc="BaoStock 逐只更新")
    for file in pbar:
        code = file.split('.')[0].split('_')[0]
        file_path = os.path.join(HIST_DATA_DIR, file)
        stock_name = stock_name_cache.get(code)
        if not stock_name: continue
            
        try:
            df_hist = pd.read_csv(file_path)
        except Exception: continue
            
        if not df_hist.empty and TODAY_DATE in df_hist['日期'].astype(str).values:
            today_data = df_hist[df_hist['日期'].astype(str) == TODAY_DATE].iloc[-1]
            daily_snapshot_list.append(today_data.to_dict())
            cache_hit_count += 1
            continue

        bs_code = f"sh.{code}" if code.startswith(('60', '688', '900')) else f"sz.{code}"
        rs = None
        data_list = []
        try:
            rs = bs.query_history_k_data_plus(bs_code, "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg", start_date=TODAY_DATE, end_date=TODAY_DATE, frequency="d", adjustflag="2")
        except Exception as e: pass

        if rs is None or rs.error_code != '0':
            try:
                bs.logout()
                bs.login()
                rs = bs.query_history_k_data_plus(bs_code, "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg", start_date=TODAY_DATE, end_date=TODAY_DATE, frequency="d", adjustflag="2")
            except Exception: continue

        if rs is None or rs.error_code != '0': continue
        while rs.next(): data_list.append(rs.get_row_data())
        if not data_list: continue

        try:
            df_today = pd.DataFrame(data_list, columns=rs.fields)
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_cols: df_today[col] = pd.to_numeric(df_today[col], errors='coerce')

            df_today['振幅'] = ((df_today['high'] - df_today['low']) / df_today['preclose'] * 100).round(2)
            df_today['涨跌额'] = (df_today['close'] - df_today['preclose']).round(2)

            today_row = {
                '日期': df_today['date'].iloc[0], '股票代码': code, '股票名称': stock_name,
                '开盘': round(df_today['open'].iloc[0], 2), '收盘': round(df_today['close'].iloc[0], 2),
                '最高': round(df_today['high'].iloc[0], 2), '最低': round(df_today['low'].iloc[0], 2),
                '成交量': df_today['volume'].iloc[0], '成交额': df_today['amount'].iloc[0],
                '振幅': df_today['振幅'].iloc[0], '涨跌幅': round(df_today['pctChg'].iloc[0], 2),
                '涨跌额': df_today['涨跌额'].iloc[0], '换手率': round(df_today['turn'].iloc[0], 4)
            }
            daily_snapshot_list.append(today_row)
            df_hist.loc[len(df_hist)] = today_row
            df_hist.to_csv(file_path, index=False, encoding='utf-8-sig')
            update_count += 1
        except Exception: continue

    bs.logout()
    print(f"\n✅ BaoStock 数据完毕！(新下载: {update_count} | 缓存命中: {cache_hit_count})")
    return pd.DataFrame(daily_snapshot_list) if daily_snapshot_list else None

def _fetch_single_news(row):
    code, name = row['股票代码'], row.get('股票名称', row['股票代码'])
    try:
        df_stock_news = ak.stock_news_em(symbol=code)
        return [{"关联代码": code, "关联名称": name, "发布时间": str(nr.get("发布时间", "")), "标题": nr.get("新闻标题", ""), "内容": nr.get("新闻内容", "")} for _, nr in df_stock_news.head(5).iterrows()] if not df_stock_news.empty else []
    except Exception: return []

def fetch_alternative_data(df_spot, TODAY_DATE, TODAY_STR):
    news_file = os.path.join(ALT_DATA_DIR, f"news_{TODAY_STR}.csv")
    if os.path.exists(news_file):
        print(f"\n⚡ 检测到今日舆情新闻已存在，极速读取本地缓存: {news_file}")
        return pd.read_csv(news_file)

    print("\n🕵️‍♂️ 启动量化情报榨取工厂 (双网捕捞模式)")
    news_list = []
    try:
        df_lhb = ak.stock_lhb_detail_em(start_date=TODAY_STR, end_date=TODAY_STR)
        if not df_lhb.empty: df_lhb.to_csv(os.path.join(ALT_DATA_DIR, f"lhb_{TODAY_STR}.csv"), index=False, encoding='utf-8-sig')
    except Exception: pass

    try:
        df_global = ak.stock_info_global_cls()
        if not df_global.empty: news_list.extend([{"关联代码": "宏观", "关联名称": "全市场", "发布时间": r.get("发布时间", TODAY_STR), "标题": r.get("标题", ""), "内容": r.get("内容", "")} for _, r in df_global.iterrows()])
    except Exception: pass

    if df_spot is not None and not df_spot.empty:
        target_stocks = df_spot[(df_spot['换手率'] > 3.0) & (abs(df_spot['涨跌幅']) > 2.0)].sort_values(by='换手率', ascending=False).head(200)
        with ThreadPoolExecutor(max_workers=10) as executor:
            for fut in tqdm(as_completed([executor.submit(_fetch_single_news, row) for _, row in target_stocks.iterrows()]), total=len(target_stocks), desc="地网抓取"):
                news_list.extend(fut.result())

    if news_list:
        df_news_final = pd.DataFrame(news_list).drop_duplicates(subset=['标题'], keep='first')
        df_news_final.to_csv(news_file, index=False, encoding='utf-8-sig')
        return df_news_final
    return None

# ==========================================
# 阶段二：多线程 AI NLP 特征捕捞
# ==========================================
def process_single_news(row, client, system_prompt, TODAY_STR):
    text = str(row.get('标题', '')) + " " + str(row.get('内容', ''))
    if len(text.strip()) < 10: return None
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"请分析以下新闻：\n{text}"}],
            temperature=0.1, timeout=25.0
        )
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            res = json.loads(match.group(0))
            res['news_time'] = row.get('发布时间', TODAY_STR)
            return res
    except Exception: pass
    return None

def batch_process_news(df_news, TODAY_STR):
    out_file = os.path.join(LLM_OUT_DIR, f"llm_news_features_{TODAY_STR}.csv")
    if os.path.exists(out_file): return pd.read_csv(out_file)

    if df_news is None or df_news.empty:
        news_file = os.path.join(ALT_DATA_DIR, f"news_{TODAY_STR}.csv")
        if not os.path.exists(news_file): return None
        df_news = pd.read_csv(news_file)

    if not GEMINI_API_KEY: return None

    client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    system_prompt = """
    【角色设定】
    你是一位拥有20年经验的华尔街顶级A股量化对冲基金经理，也是极其严谨的金融数据提取专家。
    任务：分析给定的A股财经新闻，剥离情绪化表达，提取结构化因子。
    【输出要求】纯 JSON 对象，无Markdown。
    { "sentiment_score": float, "concept": str, "is_trap": bool, "related_company": str }
    """

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for fut in tqdm(as_completed([executor.submit(process_single_news, row, client, system_prompt, TODAY_STR) for _, row in df_news.iterrows()]), total=len(df_news), desc="LLM特征提炼"):
            if fut.result(): results.append(fut.result())

    if results:
        df_features = pd.DataFrame(results)
        df_features.to_csv(out_file, index=False, encoding='utf-8-sig')
        return df_features
    return None

# ==========================================
# 阶段三：16线程 K 线特征缝合与 LGBM 实盘决策
# ==========================================
def process_llm_features(df_llm, TODAY_STR):
    if df_llm is None or df_llm.empty: return None
    df_llm = df_llm[df_llm['related_company'] != '无'].copy()
    df_llm['日期'] = df_llm['news_time'].astype(str).str.slice(0, 10)
    df_llm['related_company'] = df_llm['related_company'].str.replace('、', ',').str.replace(' ', '')
    df_llm['company_list'] = df_llm['related_company'].str.split(',')
    df_exploded = df_llm.explode('company_list').rename(columns={'company_list': '股票名称'})
    return df_exploded.groupby(['日期', '股票名称']).agg({'sentiment_score': 'mean', 'is_trap': 'max', 'concept': 'first'}).reset_index()

def _process_single_stock_history(file):
    try:
        df_stock = pd.read_csv(os.path.join(HIST_DATA_DIR, file))
        if df_stock.empty or '收盘' not in df_stock.columns: return None
        df_stock['target_return_T1'] = (df_stock['开盘'].shift(-2) / df_stock['开盘'].shift(-1)) - 1.0
        df_stock['MA5_bias'] = df_stock['收盘'] / df_stock['收盘'].rolling(5).mean() - 1.0
        df_stock['MA20_bias'] = df_stock['收盘'] / df_stock['收盘'].rolling(20).mean() - 1.0
        df_stock['Vol_Ratio'] = df_stock['成交量'] / df_stock['成交量'].rolling(5).mean()
        df_stock['Volatility_10d'] = df_stock['振幅'].rolling(10).mean()
        df_stock['Momentum_3d'] = df_stock['收盘'] / df_stock['收盘'].shift(3) - 1.0
        df_stock['Vol_Breakout'] = df_stock['成交量'] / df_stock['成交量'].rolling(10).max().shift(1)
        df_stock['Price_Breakout_Dist'] = df_stock['收盘'] / df_stock['最高'].rolling(20).max().shift(1) - 1.0
        df_stock['Upper_Shadow'] = (df_stock['最高'] - df_stock[['开盘', '收盘']].max(axis=1)) / (df_stock['最高'] - df_stock['最低'] + 0.001)
        df_stock['Penny_Stock_Reversal'] = np.where(df_stock['收盘'] <= 20.0, df_stock['Vol_Ratio'] * df_stock['换手率'], 0)
        return df_stock
    except Exception: return None

def build_machine_learning_dataset(df_llm_features, TODAY_STR):
    files = [f for f in os.listdir(HIST_DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
    if not files: return None
    
    all_data = []
    print(f"\n🧬 启动 16 线程心算高维因子 (历史样本数: {len(files)})...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        for fut in tqdm(as_completed([executor.submit(_process_single_stock_history, f) for f in files]), total=len(files), desc="K线并算"):
            if fut.result() is not None: all_data.append(fut.result())

    if not all_data: return None
    df_all_stocks = pd.concat(all_data, ignore_index=True)
    df_all_stocks['clean_name'] = df_all_stocks['股票名称'].str.replace('-U', '').str.replace('A', '')
    
    if df_llm_features is not None and not df_llm_features.empty:
        df_llm_features['clean_name'] = df_llm_features['股票名称']
        df_merged = pd.merge(df_all_stocks, df_llm_features[['日期', 'clean_name', 'sentiment_score', 'is_trap']], on=['日期', 'clean_name'], how='left')
    else:
        df_merged = df_all_stocks.copy()
        df_merged['sentiment_score'], df_merged['is_trap'] = 0.0, 0
        
    df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0.0)
    df_merged['is_trap'] = df_merged['is_trap'].fillna(False).astype(int)
    df_merged.to_csv(os.path.join(MERGED_DATA_DIR, "lgb_train_dataset.csv"), index=False, encoding='utf-8-sig')
    return df_merged

def generate_reason(row, pool_type):
    reasons = []
    if row.get('sentiment_score', 0) >= 0.7: reasons.append(f"🔥AI极强利好(分:{row['sentiment_score']:.1f})")
    elif row.get('sentiment_score', 0) >= 0.3: reasons.append(f"📰新闻呵护")

    if pool_type == "hot":
        reasons.append(f"🚀动量强劲(涨幅{row['涨跌幅']}%)")
        if row.get('Vol_Breakout', 0) > 1.5: reasons.append(f"💥爆量突破")
    elif pool_type == "steady":
        reasons.append(f"📈趋势爬升(涨幅{row['涨跌幅']}%)")
    return " | ".join(reasons) if reasons else "✨多维衍生因子交汇共振"

def train_and_predict(df):
    print("\n🧠 启动 LightGBM 训练引擎...")
    df['市值代理'] = np.where(df['换手率'] > 0, df['成交额'] / df['换手率'], 0)
    df = df.sort_values(by=['日期', '股票代码'])
    features = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '换手率', 'sentiment_score', 'is_trap', 'MA5_bias', 'MA20_bias', 'Vol_Ratio', 'Volatility_10d', 'Momentum_3d', '市值代理', 'Vol_Breakout', 'Price_Breakout_Dist', 'Upper_Shadow', 'Penny_Stock_Reversal'] 
    df[features] = df[features].fillna(0)
    
    predict_date = df['日期'].max()
    test_data = df[df['日期'] == predict_date].copy()
    train_data = df[df['日期'] < predict_date].copy().dropna(subset=['target_return_T1'])
    train_data['sample_weight'] = np.exp(-np.log(2) * (pd.to_datetime(predict_date) - pd.to_datetime(train_data['日期'])).dt.days / 60)
        
    train_set = lgb.Dataset(train_data[features], label=train_data['target_return_T1'], weight=train_data['sample_weight'])
    model = lgb.train(
        {'objective': 'huber', 'metric': 'rmse', 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1},
        train_set, num_boost_round=100
    )
    test_data['predicted_return'] = model.predict(test_data[features])
    safe_pool = test_data[test_data['is_trap'] == 0].copy()
    
    affordable_hot = safe_pool[(safe_pool['收盘'] <= 15.0) & (safe_pool['涨跌幅'] >= 7.0)].sort_values(by='predicted_return', ascending=False)
    affordable_steady = safe_pool[(safe_pool['收盘'] <= 15.0) & (safe_pool['涨跌幅'] >= 1.0) & (safe_pool['涨跌幅'] <= 6.0)].sort_values(by='predicted_return', ascending=False)
    
    def calculate_order(df_pool, total_cash, max_slots, is_hot):
        df_pool = df_pool.copy()
        df_pool['买入股数'], df_pool['实际消耗资金'] = 0, 0.0
        df_pool['T+1止盈(阻力)'] = (df_pool['收盘'] * (1 + np.maximum(df_pool['predicted_return'], 0.09 if is_hot else 0.05))).round(2)
        df_pool['T+1纠错(支撑)'] = (df_pool['收盘'] * (0.94 if is_hot else 0.96)).round(2)
        
        rem_cash, slots = total_cash, 0
        for idx, row in df_pool.iterrows():
            if slots >= max_slots or rem_cash < 300: break
            shares = int((total_cash / max_slots) // (row['收盘'] * 100)) * 100
            if shares == 0 and rem_cash >= row['收盘'] * 100: shares = 100
            while shares * row['收盘'] > rem_cash and shares > 0: shares -= 100
            if shares > 0:
                df_pool.at[idx, '买入股数'] = shares
                df_pool.at[idx, '实际消耗资金'] = shares * row['收盘']
                rem_cash -= shares * row['收盘']
                slots += 1
                
        res = df_pool.head(max_slots + 2).copy()
        res['执行时机'] = res.apply(lambda r: "【选定打靶】" if r['买入股数']>0 else "【备选观察】", axis=1)
        res['买入理由'] = res.apply(lambda r: generate_reason(r, "hot" if is_hot else "steady"), axis=1)
        return res
        
    _print_dashboard(calculate_order(affordable_hot, HOT_CAPITAL, HOT_SLOTS, True), calculate_order(affordable_steady, STEADY_CAPITAL, STEADY_SLOTS, False), predict_date)
    model.save_model(os.path.join(MODEL_DIR, "lgbm_master_v1.txt")) 

def _print_dashboard(hot, steady, date):
    def fmt(df): return df[['股票代码', 'clean_name', '收盘', '买入股数', '执行时机', 'T+1止盈(阻力)', 'T+1纠错(支撑)', '买入理由']].rename(columns={'股票代码':'代码', 'clean_name':'名称'})
    print(f"\n🚀 A股量化实盘系统 - {date} 终极买卖指令单 🚀\n")
    print("⚔️ 游资突击队:\n", fmt(hot).to_string(index=False) if not hot.empty else "无标的")
    print("\n🛡️ 稳健主力池:\n", fmt(steady).to_string(index=False) if not steady.empty else "无标的\n")

if __name__ == "__main__":
    start_time = time.time()
    init_environment()
    
    with BypassProxyContext():
        latest_str, latest_date = get_latest_trade_date()
        print(f"📅 今日交易日为：{latest_date}")
        is_macro_risk = check_macro_risk(latest_date)
        spot_data = update_daily_price_with_baostock(latest_date, latest_str)
        news_data = fetch_alternative_data(spot_data, latest_date, latest_str) if spot_data is not None else None
            
    llm_features = batch_process_news(news_data, latest_str)
    df_llm_cleaned = process_llm_features(llm_features, latest_str) if llm_features is not None else None
    df_dataset = build_machine_learning_dataset(df_llm_cleaned, latest_str)
    
    if df_dataset is not None and not df_dataset.empty: 
        if is_macro_risk: print("🚨 宏观风控熔断，今日强制空仓！")
        else: train_and_predict(df_dataset)
    else: print("❌ 数据集为空，无法执行训练！")
    
    print(f"\n⏱️ 全系统处理流闭环完成。总耗时: {time.time() - start_time:.1f} 秒。")