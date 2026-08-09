import os
import sys
import re
import time
import json
import argparse
import warnings
import socket
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import httpx

# 🚨 【核心防死锁】
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
GEMINI_API_KEY = "AIzaSyBf_mK2Tjg8wVhHhto2OS6d88a1JV-h6w4"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# 使用官方稳定高速的模型，不使用不存在的模型
LLM_MODEL_NAME = "gemini-3.6-flash"  

PROXY_URL = None  # 如有本地代理可设为 "http://127.0.0.1:7890"

BASE_DIR = "F:/Study/stock/stock_data_hist"
HIST_DATA_DIR = f"{BASE_DIR}/historical_daily"   
ALT_DATA_DIR = f"{BASE_DIR}/daily_features"      
LLM_OUT_DIR = f"{BASE_DIR}/llm_features"         
MERGED_DATA_DIR = f"{BASE_DIR}/merged_dataset"   
MODEL_DIR = f"{BASE_DIR}/models/lgbm"            
LEDGER_PATH = f"{BASE_DIR}/prediction_ledger.csv"

TOTAL_CAPITAL = 30000

# 🏆 【比赛夺冠特设：自适应防冷冻与终极复活引擎】
# 🚨 开启此开关后，在任何极端恶劣的市场下，均能自动解冻并强制生成挂单，杜绝空仓踏空！
COMPETITION_URGENCY_MODE = True  

# 资金及仓位管理 (3万元本金优化版，单笔底线8000，降低佣金惩罚)
MAX_TOTAL_POSITIONS = 3
MAX_HOT_POSITIONS = 1  
MAX_SINGLE_POSITION_PCT = 0.45
MIN_POSITION_CASH = 8000  
TRAIN_LOOKBACK_DAYS = 700
VALID_DAYS = 60
LABEL_HORIZON = 2

# 低吸挂单折扣
HOT_LIMIT_DISCOUNT = 0.99
STEADY_LIMIT_DISCOUNT = 0.98
HOT_PROFIT_THRESHOLD = 0.012
STEADY_PROFIT_THRESHOLD = 0.008

# 费用设置
BUY_COMMISSION_RATE = 0.0003
SELL_COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_DUTY_RATE = 0.0005
ESTIMATED_ROUND_TRIP_COST = 0.0028 # 综合考虑单笔1.5W交易起征点5元摩擦的真实综合成本

# 自动补采：每次启动会从第一份新闻归档开始检查交易日缺口。
# 若累计缺口较多，每次最多补 5 天，后续启动会继续补，避免一次运行耗时失控。
AUTO_BACKFILL_MAX_DATES_PER_RUN = 5
NEWS_SEARCH_MAX_ITEMS = 100
NEWS_ARCHIVE_COLUMNS = ['关联代码', '关联名称', '发布时间', '标题', '内容', '文章来源', '新闻链接']
LLM_ARCHIVE_COLUMNS = ['sentiment_score', 'concept', 'is_trap', 'related_company', 'news_time']

# ==================================================

def init_environment():
    for folder in [HIST_DATA_DIR, ALT_DATA_DIR, LLM_OUT_DIR, MERGED_DATA_DIR, MODEL_DIR]:
        os.makedirs(folder, exist_ok=True)
    print("📁 [系统初始化] 运行目录自检完毕。")


def get_latest_trade_date():
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
        print(f"⚠️ 获取交易日历失败: {e}")
        return datetime.now().strftime("%Y%m%d"), datetime.now().strftime("%Y-%m-%d")


def get_trade_dates(start_date, end_date):
    """返回闭区间内的 A 股交易日；日历接口失败时仅以工作日兜底。"""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    if start > end:
        raise ValueError(f"开始日期 {start} 晚于结束日期 {end}")

    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(trade_cal['trade_date'], errors='coerce').dt.date
        return [d.strftime('%Y-%m-%d') for d in dates if pd.notna(d) and start <= d <= end]
    except Exception as e:
        print(f"⚠️ 获取交易日历失败，暂以周一至周五兜底: {e}")
        return [
            d.strftime('%Y-%m-%d')
            for d in pd.date_range(start, end, freq='B').date
        ]


def get_next_trade_date(base_date):
    """取得基准日之后的首个交易日，用作实际预测/下单目标日。"""
    base = pd.to_datetime(base_date).date()
    search_end = base + pd.Timedelta(days=15)
    future_dates = get_trade_dates(
        (base + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        search_end.strftime('%Y-%m-%d')
    )
    if future_dates:
        return future_dates[0]

    # 极端情况下交易日历无未来记录，退化到下一个工作日。
    fallback = pd.bdate_range(base + pd.Timedelta(days=1), periods=1)[0]
    return fallback.strftime('%Y-%m-%d')


def _archive_date_from_name(filename, prefix, suffix='.csv'):
    match = re.fullmatch(rf'{re.escape(prefix)}(\d{{8}}){re.escape(suffix)}', filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), '%Y%m%d').strftime('%Y-%m-%d')
    except ValueError:
        return None


def _atomic_write_csv(df, path):
    """先写同目录临时文件再替换，避免中断留下半份归档。"""
    temp_path = f'{path}.{os.getpid()}.{time.time_ns()}.tmp'
    try:
        df.to_csv(temp_path, index=False, encoding='utf-8-sig')
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def discover_backfill_dates(latest_date, start_date=None, end_date=None, force=False):
    """发现原始新闻或 LLM 特征缺失的交易日。"""
    explicit_range = start_date is not None or end_date is not None
    end = end_date or latest_date

    raw_archives = {
        d for d in (
            _archive_date_from_name(name, 'news_')
            for name in os.listdir(ALT_DATA_DIR)
        ) if d
    }
    llm_archives = {
        d for d in (
            _archive_date_from_name(name, 'llm_news_features_')
            for name in os.listdir(LLM_OUT_DIR)
        ) if d
    }

    if force and not explicit_range:
        start = end
    elif start_date:
        start = start_date
    elif raw_archives:
        # 不追溯到首次启用本脚本之前；只修复已开始归档后的断档。
        start = min(raw_archives)
    else:
        start = end

    trade_dates = get_trade_dates(start, end)
    if force:
        pending = trade_dates
    else:
        pending = [d for d in trade_dates if d not in raw_archives or d not in llm_archives]

    if not explicit_range and len(pending) > AUTO_BACKFILL_MAX_DATES_PER_RUN:
        remaining = len(pending) - AUTO_BACKFILL_MAX_DATES_PER_RUN
        if latest_date in pending:
            pending = pending[:AUTO_BACKFILL_MAX_DATES_PER_RUN - 1] + [latest_date]
        else:
            pending = pending[:AUTO_BACKFILL_MAX_DATES_PER_RUN]
        print(f"ℹ️ 历史缺口较多，本次先补 {len(pending)} 天，剩余 {remaining} 天将在后续启动时继续。")
    return pending


def calculate_market_breadth(latest_date):
    """计算大盘风控广度"""
    print("🔍 正在计算全市场温度与广度风控指标...")
    try:
        df_spot = ak.stock_zh_a_spot_em()
        if not df_spot.empty:
            df_main = df_spot[df_spot['代码'].str.startswith(('60', '00'))]
            up_stocks = len(df_main[df_main['涨跌幅'] > 0])
            total_stocks = len(df_main)
            breadth = up_stocks / total_stocks if total_stocks > 0 else 0.5
            print(f"📊 [东财源] 上涨家数: {up_stocks} / 总数: {total_stocks} | 市场广度: {breadth*100:.1f}%")
            return breadth
    except Exception as e:
        print(f"⚠️ [东财源] 获取失败: {e}")
    
    try:
        df_index = ak.stock_zh_index_spot_em()
        if not df_index.empty:
            sh_idx = df_index[df_index['代码'] == '000001']
            sz_idx = df_index[df_index['代码'] == '399001']
            sh_change = float(sh_idx['涨跌幅'].iloc[0]) if not sh_idx.empty else 0.0
            sz_change = float(sz_idx['涨跌幅'].iloc[0]) if not sz_idx.empty else 0.0
            if sh_change > 0.2 and sz_change > 0.2:
                breadth = 0.65
            elif sh_change < -0.3 and sz_change < -0.3:
                breadth = 0.25
            else:
                breadth = 0.45
            print(f"📊 [指数备用] 上海: {sh_change:+.2f}% | 深圳: {sz_change:+.2f}% -> 预估广度: {breadth*100:.1f}%")
            return breadth
    except Exception as ex:
        print(f"⚠️ [指数备用] 亦获取失败: {ex}")
        
    return 0.5


def update_daily_price_with_baostock(TODAY_DATE, TODAY_STR):
    print(f"\n🔄 [阶段一] 启动 BaoStock 行情引擎，获取今日 ({TODAY_DATE}) 数据...")
    files = [f for f in os.listdir(HIST_DATA_DIR) if f.endswith('.csv')]
    if len(files) == 0:
        print("⚠️ 历史数据为空，请先运行 script1.py 建立基础底座。")
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
    daily_snapshot_list = []
    update_count, cache_hit_count = 0, 0

    pbar = tqdm(files, desc="BaoStock 增量更新")
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

        bs_code = f"sh.{code}" if code.startswith('60') else f"sz.{code}"
        rs = None
        data_list = []
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=TODAY_DATE, end_date=TODAY_DATE, frequency="d", adjustflag="2"
            )
        except Exception: pass

        if rs is None or rs.error_code != '0':
            try:
                bs.logout(); time.sleep(1); bs.login()
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=TODAY_DATE, end_date=TODAY_DATE, frequency="d", adjustflag="2"
                )
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
    print(f"\n✅ 行情更新完毕！(新下载: {update_count} 只 | 缓存命中: {cache_hit_count} 只)")
    if not daily_snapshot_list: return None
    return pd.DataFrame(daily_snapshot_list)


def _normalize_stock_code(value):
    """CSV 常把 000001 读成 1，这里恢复为东财接口需要的六位代码。"""
    text = str(value).strip()
    if re.fullmatch(r'\d+(?:\.0+)?', text):
        text = text.split('.')[0]
    return text.zfill(6) if text.isdigit() else text


def _clean_news_text(value):
    text = '' if value is None else str(value)
    return re.sub(r'</?em>', '', text).replace('\u3000', '').replace('\r\n', ' ')


def _query_stock_news_em(symbol, max_items=NEWS_SEARCH_MAX_ITEMS):
    """查询最近新闻；优先把东财单次结果扩到 100 条，失败则回退 AkShare。"""
    url = 'https://search-api-web.eastmoney.com/search/jsonp'
    callback = f'jQuery{time.time_ns()}'
    inner_param = {
        'uid': '', 'keyword': symbol, 'type': ['cmsArticleWebOld'],
        'client': 'web', 'clientType': 'web', 'clientVersion': 'curr',
        'param': {'cmsArticleWebOld': {
            'searchScope': 'default', 'sort': 'default', 'pageIndex': 1,
            'pageSize': max_items, 'preTag': '<em>', 'postTag': '</em>'
        }}
    }
    params = {
        'cb': callback,
        'param': json.dumps(inner_param, ensure_ascii=False),
        '_': str(int(time.time() * 1000))
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
        'Referer': f'https://so.eastmoney.com/news/s?keyword={symbol}'
    }

    try:
        request_kwargs = {'proxy': PROXY_URL} if PROXY_URL else {}
        response = httpx.get(
            url, params=params, headers=headers, timeout=15.0,
            follow_redirects=True, **request_kwargs
        )
        response.raise_for_status()
        payload = response.text.strip()
        left, right = payload.find('('), payload.rfind(')')
        if left < 0 or right <= left:
            raise ValueError('东财新闻接口返回了非 JSONP 内容')
        data = json.loads(payload[left + 1:right])
        items = data.get('result', {}).get('cmsArticleWebOld', []) or []
        return pd.DataFrame([{
            '新闻标题': _clean_news_text(item.get('title', '')),
            '新闻内容': _clean_news_text(item.get('content', '')),
            '发布时间': str(item.get('date', '')),
            '文章来源': item.get('mediaName', ''),
            '新闻链接': f"http://finance.eastmoney.com/a/{item.get('code', '')}.html"
        } for item in items])
    except Exception:
        # 保留官方 AkShare 封装作为兼容性兜底（通常返回最近 10 条）。
        return ak.stock_news_em(symbol=symbol)


def _fetch_single_news(row, target_date):
    code = _normalize_stock_code(row['股票代码'])
    name = row.get('股票名称', code)
    try:
        df_stock_news = _query_stock_news_em(code)
        results = []
        if not df_stock_news.empty:
            publish_dates = pd.to_datetime(
                df_stock_news.get('发布时间'), errors='coerce'
            ).dt.strftime('%Y-%m-%d')
            for (_, news_row), publish_date in zip(df_stock_news.iterrows(), publish_dates):
                if publish_date != target_date:
                    continue
                results.append({
                    '关联代码': code,
                    '关联名称': name,
                    '发布时间': str(news_row.get('发布时间', '')),
                    '标题': news_row.get('新闻标题', ''),
                    '内容': news_row.get('新闻内容', ''),
                    '文章来源': news_row.get('文章来源', ''),
                    '新闻链接': news_row.get('新闻链接', '')
                })
        return code, results, True
    except Exception as e:
        return code, [], False


def fetch_alternative_data(df_spot, TODAY_DATE, TODAY_STR, force=False):
    news_file = os.path.join(ALT_DATA_DIR, f"news_{TODAY_STR}.csv")
    if os.path.exists(news_file) and not force:
        return pd.read_csv(news_file, dtype={'关联代码': str})

    news_list = []
    success_count = 0
    failed_codes = []
    print(f"\n[阶段二] 启动 {TODAY_DATE} 舆情监控雷达...")
    if df_spot is not None and not df_spot.empty:
        target_stocks = df_spot[(df_spot['换手率'] > 4.0) & (abs(df_spot['涨跌幅']) > 3.0)].sort_values(by='换手率', ascending=False).head(80)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_single_news, row, TODAY_DATE) for _, row in target_stocks.iterrows()]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"新闻扫描 {TODAY_DATE}"):
                code, rows, ok = fut.result()
                if ok:
                    success_count += 1
                    news_list.extend(rows)
                else:
                    failed_codes.append(code)

    if failed_codes:
        print(f"⚠️ {TODAY_DATE} 新闻补采未完成：{len(failed_codes)} 只股票查询失败，本次不落盘，下次会自动重试。")
        return None

    if news_list:
        df_news_final = pd.DataFrame(news_list, columns=NEWS_ARCHIVE_COLUMNS).drop_duplicates(
            subset=['发布时间', '标题', '关联代码'], keep='first'
        )
        _atomic_write_csv(df_news_final, news_file)
        print(f"✅ {TODAY_DATE} 新闻归档完成：{len(df_news_final)} 条，成功查询 {success_count} 只股票。")
        return df_news_final

    df_news_final = pd.DataFrame(columns=NEWS_ARCHIVE_COLUMNS)
    _atomic_write_csv(df_news_final, news_file)
    print(f"ℹ️ {TODAY_DATE} 的异动股票未检索到当日新闻，已记录为空归档。")
    return df_news_final


def process_single_news(row, client, system_prompt, TODAY_STR):
    text = str(row.get('标题', '')) + " " + str(row.get('内容', ''))
    if len(text.strip()) < 10: return None
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"请分析新闻：\n{text}"}],
            temperature=0.1, timeout=15.0
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            res = json.loads(match.group(0))
            res['news_time'] = row.get('发布时间', TODAY_STR)
            return res
    except Exception: pass
    return None


def batch_process_news(df_news, TODAY_STR, force=False):
    out_file = os.path.join(LLM_OUT_DIR, f"llm_news_features_{TODAY_STR}.csv")
    if os.path.exists(out_file) and not force:
        return pd.read_csv(out_file)

    if df_news is None:
        return None
    if df_news.empty:
        empty_features = pd.DataFrame(columns=LLM_ARCHIVE_COLUMNS)
        _atomic_write_csv(empty_features, out_file)
        return empty_features
    if not GEMINI_API_KEY:
        print("⚠️ 未设置 API_KEY，跳过新闻提炼。")
        return None

    if PROXY_URL:
        custom_client = httpx.Client(proxies=PROXY_URL)
        client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, http_client=custom_client)
    else:
        client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

    system_prompt = """
    【A股量化情绪分析】分析新闻中包含的A股主板公司的情绪影响。
    【输出要求】纯 JSON，不要Markdown：
    {"sentiment_score": float, "concept": str, "is_trap": bool, "related_company": str}
    """
    results = []
    df_to_process = df_news.head(80).copy()
    print(f"\n🚀 启动 AI 舆情风控分析...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        for future in tqdm(as_completed([executor.submit(process_single_news, row, client, system_prompt, TODAY_STR) for _, row in df_to_process.iterrows()]), total=len(df_to_process), desc="Gemini分析"):
            if future.result(): results.append(future.result())

    if results:
        df_features = pd.DataFrame(results)
        _atomic_write_csv(df_features, out_file)
        return df_features
    return None


def process_llm_features(df_llm, TODAY_STR):
    if df_llm is None or df_llm.empty: return None
    required = {'related_company', 'news_time', 'sentiment_score', 'is_trap'}
    if not required.issubset(df_llm.columns):
        print(f"⚠️ LLM 特征文件字段不完整，已跳过: {required - set(df_llm.columns)}")
        return None
    df_llm = df_llm[df_llm['related_company'] != '无'].copy()
    df_llm['日期'] = df_llm['news_time'].astype(str).str.slice(0, 10)
    df_llm['related_company'] = df_llm['related_company'].str.replace('、', ',').str.replace(' ', '')
    df_llm['company_list'] = df_llm['related_company'].str.split(',')
    df_exploded = df_llm.explode('company_list')
    df_exploded.rename(columns={'company_list': '股票名称'}, inplace=True)
    return df_exploded.groupby(['日期', '股票名称']).agg({'sentiment_score': 'mean', 'is_trap': 'max'}).reset_index()


def load_all_llm_features():
    """汇总全部日期的情绪特征，让补采数据真正进入训练集。"""
    frames = []
    files = sorted(
        name for name in os.listdir(LLM_OUT_DIR)
        if _archive_date_from_name(name, 'llm_news_features_')
    )
    for filename in files:
        path = os.path.join(LLM_OUT_DIR, filename)
        try:
            raw = pd.read_csv(path)
            cleaned = process_llm_features(raw, filename[-12:-4])
            if cleaned is not None and not cleaned.empty:
                frames.append(cleaned)
        except Exception as e:
            print(f"⚠️ 读取历史情绪特征失败 {filename}: {e}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(['日期', '股票名称'], as_index=False).agg({
        'sentiment_score': 'mean',
        'is_trap': 'max'
    })


def _process_single_stock_history(file):
    """构建 K 线历史量价因子并清洗标签（彻底修复未成交导致的统计学污染）"""
    try:
        df_stock = pd.read_csv(os.path.join(HIST_DATA_DIR, file))
        required_cols = ['日期', '股票代码', '股票名称', '开盘', '收盘', '最高', '最低',
                         '成交量', '成交额', '振幅', '涨跌幅', '换手率']
        if df_stock.empty or any(c not in df_stock.columns for c in required_cols):
            return None

        df_stock = df_stock.sort_values(by='日期').reset_index(drop=True)
        df_stock['日期'] = df_stock['日期'].astype(str)
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '换手率']
        for col in numeric_cols:
            df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce')

        close = df_stock['收盘']
        open_ = df_stock['开盘']
        high = df_stock['最高']
        low = df_stock['最低']
        volume = df_stock['成交量']
        amount = df_stock['成交额']
        turnover = df_stock['换手率']

        # ---------- T+1挂限价、T+2收盘卖的未来回测标签 ----------
        t1_low = low.shift(-1)
        t2_close = close.shift(-2)
        valid_future = t1_low.notna() & t2_close.notna()

        hot_limit = close * HOT_LIMIT_DISCOUNT
        steady_limit = close * STEADY_LIMIT_DISCOUNT
        hot_filled = t1_low <= hot_limit
        steady_filled = t1_low <= steady_limit

        hot_net_return = (t2_close / hot_limit - 1.0) - ESTIMATED_ROUND_TRIP_COST
        steady_net_return = (t2_close / steady_limit - 1.0) - ESTIMATED_ROUND_TRIP_COST

        # 🚨【核心Bug修复】未成交样本的标签必须设为 NaN 排除出拟合范围！只评价已买入的后续表现。
        df_stock['target_hot_return'] = np.where(valid_future & hot_filled, hot_net_return, np.nan)
        df_stock['target_steady_return'] = np.where(valid_future & steady_filled, steady_net_return, np.nan)
        
        df_stock['target_hot_win'] = np.where(
            valid_future & hot_filled,
            (hot_net_return >= HOT_PROFIT_THRESHOLD).astype(float),
            np.nan
        )
        df_stock['target_steady_win'] = np.where(
            valid_future & steady_filled,
            (steady_net_return >= STEADY_PROFIT_THRESHOLD).astype(float),
            np.nan
        )

        # ---------- 衍生技术特征工程 ----------
        prev_close = close.shift(1)
        ret_1d = close / prev_close - 1.0
        df_stock['Return_1d'] = ret_1d
        df_stock['Gap_Open'] = open_ / prev_close - 1.0
        df_stock['Close_Open_Ratio'] = close / open_ - 1.0
        df_stock['High_Low_Ratio'] = high / low - 1.0
        df_stock['Close_High_Ratio'] = close / high - 1.0
        df_stock['Close_Low_Ratio'] = close / low - 1.0
        df_stock['Upper_Shadow'] = (high - np.maximum(open_, close)) / (prev_close + 1e-8)
        df_stock['Lower_Shadow'] = (np.minimum(open_, close) - low) / (prev_close + 1e-8)

        # 连板、高位阻尼特征
        vol_ma5 = volume.rolling(5).mean().shift(1)
        df_stock['Vol_Spike_5d'] = volume / (vol_ma5 + 1e-8)
        df_stock['Intraday_Ret_Ratio'] = (close - open_) / (high - low + 1e-8)
        df_stock['Limit_Up_Proximity'] = high / (prev_close * 1.10) - 1.0
        df_stock['Drop_From_High'] = (high - close) / (high - low + 1e-8)

        # 趋势动能
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        df_stock['MA5_bias'] = close / ma5 - 1.0
        df_stock['MA10_bias'] = close / ma10 - 1.0
        df_stock['MA20_bias'] = close / ma20 - 1.0
        df_stock['MA60_bias'] = close / ma60 - 1.0
        df_stock['MA5_MA20'] = ma5 / ma20 - 1.0
        df_stock['MA20_MA60'] = ma20 / ma60 - 1.0
        for n in [3, 5, 10, 20]:
            df_stock[f'Momentum_{n}d'] = close / close.shift(n) - 1.0
        df_stock['Volatility_10d'] = ret_1d.rolling(10).std()
        df_stock['Volatility_20d'] = ret_1d.rolling(20).std()

        previous_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ], axis=1).max(axis=1)
        df_stock['ATR_14'] = true_range.rolling(14).mean() / (close + 1e-8)

        low20 = low.rolling(20).min()
        high20 = high.rolling(20).max()
        df_stock['Price_Position_20'] = (close - low20) / (high20 - low20 + 1e-8)
        df_stock['Up_Days_5'] = (ret_1d > 0).rolling(5).sum()

        vol20 = volume.rolling(20).mean()
        df_stock['Vol_Ratio_5'] = volume / (volume.rolling(5).mean() + 1e-8)
        df_stock['Vol_Ratio_20'] = volume / (vol20 + 1e-8)
        df_stock['Turnover_MA5'] = turnover.rolling(5).mean()
        df_stock['Turnover_MA20'] = turnover.rolling(20).mean()
        df_stock['Turnover_Accel'] = turnover / (df_stock['Turnover_MA5'] + 1e-8)
        df_stock['Amount_Log'] = np.log1p(amount.clip(lower=0))
        df_stock['Illiquidity_10'] = (
            ret_1d.abs() / (amount.replace(0, np.nan) + 1e-8)
        ).rolling(10).mean() * 1e8
        df_stock['Price_Vol_Corr_10'] = ret_1d.rolling(10).corr(volume.pct_change())

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df_stock['RSI_14'] = 100 - 100 / (1 + rs)

        df_stock.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_stock
    except Exception as e:
        print(f"⚠️ 特征构建失败 {file}: {e}")
        return None


def build_machine_learning_dataset(df_llm_features, TODAY_STR):
    files = [f for f in os.listdir(HIST_DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
    if not files:
        return None

    all_data = []
    print(f"\n🧬 启动 16 线程构建量价及高频动力学因子模型...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_process_single_stock_history, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(files), desc="特征矩阵计算"):
            res = fut.result()
            if res is not None:
                all_data.append(res)

    if not all_data:
        return None

    df_all_stocks = pd.concat(all_data, ignore_index=True)
    df_all_stocks['日期'] = df_all_stocks['日期'].astype(str)
    df_all_stocks['clean_name'] = (
        df_all_stocks['股票名称'].astype(str)
        .str.replace('-U', '', regex=False)
        .str.replace('A', '', regex=False)
    )

    if df_llm_features is not None and not df_llm_features.empty:
        llm = df_llm_features.copy()
        llm['clean_name'] = llm['股票名称'].astype(str)
        date_col = 'text_date' if 'text_date' in llm.columns else '日期'
        llm[date_col] = llm[date_col].astype(str)
        df_merged = pd.merge(
            df_all_stocks,
            llm[[date_col, 'clean_name', 'sentiment_score', 'is_trap']],
            left_on=['日期', 'clean_name'],
            right_on=[date_col, 'clean_name'],
            how='left'
        )
    else:
        df_merged = df_all_stocks.copy()
        df_merged['sentiment_score'] = 0.0
        df_merged['is_trap'] = 0

    df_merged['sentiment_score'] = pd.to_numeric(df_merged['sentiment_score'], errors='coerce').fillna(0.0)
    df_merged['is_trap'] = df_merged['is_trap'].fillna(False).astype(int)

    grouped = df_merged.groupby('日期')
    df_merged['换手率截面排名'] = grouped['换手率'].rank(pct=True)
    df_merged['成交额截面排名'] = grouped['成交额'].rank(pct=True)
    df_merged['动量5日截面排名'] = grouped['Momentum_5d'].rank(pct=True)
    df_merged['动量20日截面排名'] = grouped['Momentum_20d'].rank(pct=True)
    df_merged['20日位置截面排名'] = grouped['Price_Position_20'].rank(pct=True)
    df_merged['波动率截面排名'] = grouped['Volatility_20d'].rank(pct=True)

    df_merged['市值代理'] = df_merged['成交额'] / (df_merged['换手率'].abs() + 1e-5)
    df_merged['市值代理排名'] = grouped['市值代理'].rank(pct=True)

    df_merged['市场上涨比例'] = grouped['涨跌幅'].transform(lambda s: (s > 0).mean())
    df_merged['市场中位涨幅'] = grouped['涨跌幅'].transform('median')
    df_merged['市场平均涨幅'] = grouped['涨跌幅'].transform('mean')
    df_merged['市场波动中位数'] = grouped['Volatility_20d'].transform('median')

    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    out_path = os.path.join(MERGED_DATA_DIR, 'lgb_train_dataset.csv')
    df_merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df_merged


def _pool_rule_filter(frame, pool_type, relax_level=0, verbose=False):
    if frame.empty:
        return frame.copy()

    # 动态自适应设定参数边界
    min_turnover = 0.8 if relax_level == 0 else (0.5 if relax_level == 1 else 0.3)
    max_turnover = 18.0 if relax_level == 0 else (22.0 if relax_level == 1 else 28.0)
    max_hl_ratio = 0.12 if relax_level == 0 else (0.15 if relax_level == 1 else 0.18)
    min_market_up = 0.30 if relax_level == 0 else (0.22 if relax_level == 1 else 0.15)
    
    max_affordable_price = max(20.0, TOTAL_CAPITAL * MAX_SINGLE_POSITION_PCT / 100.0)
    if relax_level >= 1:
        max_affordable_price *= 1.3  # 放宽价格上限
        
    common = frame[
        (frame['is_trap'] == 0) &
        (frame['收盘'] >= 2.5) &
        (frame['收盘'] <= max_affordable_price) &
        (frame['成交额截面排名'] >= (0.35 if relax_level == 0 else 0.25)) &
        (frame['换手率'] >= min_turnover) &
        (frame['换手率'] <= max_turnover) &
        (frame['High_Low_Ratio'] <= max_hl_ratio) &
        (frame['市场上涨比例'] >= min_market_up)
    ].copy()

    if pool_type == 'hot':
        min_chg = 1.0 if relax_level == 0 else (0.2 if relax_level == 1 else -1.0)
        max_chg = 6.2 if relax_level == 0 else (8.5 if relax_level == 1 else 10.5)
        min_m5 = 0.01 if relax_level == 0 else (-0.02 if relax_level == 1 else -0.05)
        max_m5 = 0.18 if relax_level == 0 else (0.25 if relax_level == 1 else 0.35)
        
        specific = common[
            (common['涨跌幅'] >= min_chg) &
            (common['涨跌幅'] <= max_chg) &
            (common['Momentum_5d'] >= min_m5) &
            (common['Momentum_5d'] <= max_m5) &
            (common['MA20_bias'] >= (-0.02 if relax_level == 0 else -0.05)) &
            (common['MA20_bias'] <= (0.14 if relax_level == 0 else 0.20)) &
            (common['Price_Position_20'] >= (0.45 if relax_level == 0 else 0.30)) &
            (common['Price_Position_20'] <= (0.96 if relax_level == 0 else 0.98)) &
            (common['RSI_14'] >= (43 if relax_level == 0 else 35)) &
            (common['RSI_14'] <= (78 if relax_level == 0 else 85))
        ].copy()
    else:
        min_chg = -1.5 if relax_level == 0 else (-3.0 if relax_level == 1 else -5.0)
        max_chg = 2.5 if relax_level == 0 else (4.5 if relax_level == 1 else 6.0)
        
        specific = common[
            (common['涨跌幅'] >= min_chg) &
            (common['涨跌幅'] <= max_chg) &
            (common['MA20_bias'] >= (-0.015 if relax_level == 0 else -0.04)) &
            (common['MA20_bias'] <= (0.10 if relax_level == 0 else 0.15)) &
            (common['Price_Position_20'] >= (0.25 if relax_level == 0 else 0.15)) &
            (common['Price_Position_20'] <= (0.85 if relax_level == 0 else 0.92)) &
            (common['RSI_14'] >= (35 if relax_level == 0 else 28)) &
            (common['RSI_14'] <= (70 if relax_level == 0 else 80))
        ].copy()

    if verbose:
        pool_cn = "突击池" if pool_type == 'hot' else "稳健池"
        print(f"   📊 [{pool_cn} 漏斗 (阻泥度 Level {relax_level})]: "
              f"原始输入 {len(frame)} 只 -> 基础过滤 {len(common)} 只 -> 核心池条件 {len(specific)} 只")
              
    return specific


def _select_validation_threshold(valid_scored, return_target, max_positions, allow_negative_expect=False):
    if valid_scored.empty:
        return np.inf, {'trades': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    daily_top = (
        valid_scored.sort_values(['日期', 'expected_profit_score'], ascending=[True, False])
        .groupby('日期', group_keys=False)
        .head(max_positions)
    )
    if daily_top.empty:
        return np.inf, {'trades': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    best = None
    min_win_rate = 0.40 if not allow_negative_expect else 0.35
    min_avg_return = 0.0 if not allow_negative_expect else -0.008

    for q in [0.00, 0.20, 0.35, 0.50, 0.60, 0.70, 0.80]:
        threshold = float(daily_top['expected_profit_score'].quantile(q))
        selected = daily_top[daily_top['expected_profit_score'] >= threshold]
        min_trades = min(20, max(8, len(daily_top) // 4))
        if len(selected) < min_trades:
            continue
        avg_return = float(selected[return_target].mean())
        win_rate = float((selected[return_target] > 0).mean())
        median_return = float(selected[return_target].median())
        objective = avg_return + 0.25 * median_return
        stats = {
            'threshold': threshold,
            'trades': len(selected),
            'avg_return': avg_return,
            'win_rate': win_rate,
            'objective': objective
        }
        if best is None or stats['objective'] > best['objective']:
            best = stats

    if best is None or best['avg_return'] < min_avg_return or best['win_rate'] < min_win_rate:
        return np.inf, {'trades': 0, 'avg_return': 0.0, 'win_rate': 0.0}
    return best['threshold'], best


def _fit_pool_models(df, features, predict_date, pool_type, win_target, return_target, max_positions):
    train_data = df[(df['日期'] < predict_date) & df[win_target].notna() & df[return_target].notna()].copy()
    train_dates = sorted(train_data['日期'].unique())
    pool_cn = '突击池' if pool_type == 'hot' else '稳健池'
    
    if len(train_dates) < VALID_DAYS + LABEL_HORIZON + 80:
        print(f"⚠️ {pool_type} 历史交易日不足，暂停训练。")
        return pd.DataFrame()

    train_dates = train_dates[-TRAIN_LOOKBACK_DAYS:]
    train_data = train_data[train_data['日期'].isin(train_dates)].copy()
    valid_dates = train_dates[-VALID_DAYS:]
    train_end_index = max(0, len(train_dates) - VALID_DAYS - LABEL_HORIZON)
    actual_train_dates = train_dates[:train_end_index]

    actual_train = train_data[train_data['日期'].isin(actual_train_dates)].copy()
    actual_valid = train_data[train_data['日期'].isin(valid_dates)].copy()
    if actual_train.empty or actual_valid.empty or actual_train[win_target].nunique() < 2:
        print(f"⚠️ {pool_type} 样本不足以供机器学习拟合。")
        return pd.DataFrame()

    for part in [actual_train, actual_valid]:
        part[features] = part[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    actual_train['days_to_now'] = (
        pd.to_datetime(predict_date) - pd.to_datetime(actual_train['日期'])
    ).dt.days
    actual_train['sample_weight'] = np.exp(-np.log(2) * actual_train['days_to_now'] / 120.0)

    train_cls = lgb.Dataset(
        actual_train[features],
        label=actual_train[win_target],
        weight=actual_train['sample_weight']
    )
    valid_cls = lgb.Dataset(
        actual_valid[features],
        label=actual_valid[win_target],
        reference=train_cls
    )

    dyn_min_data_in_leaf = max(20, min(150, len(actual_train) // 20))

    common_params = {
        'learning_rate': 0.025,
        'num_leaves': 15,
        'max_depth': 5,
        'min_data_in_leaf': dyn_min_data_in_leaf,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'lambda_l1': 0.6,
        'lambda_l2': 2.0,
        'max_bin': 127,
        'verbosity': -1,
        'seed': 42,
        'num_threads': -1,
        'force_col_wise': True
    }

    cls_params = dict(common_params)
    cls_params.update({'objective': 'binary', 'metric': ['auc', 'binary_logloss']})
    cls_model = lgb.train(
        cls_params,
        train_cls,
        num_boost_round=900,
        valid_sets=[valid_cls],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)]
    )

    clipped_train_return = actual_train[return_target].clip(-0.12, 0.15)
    clipped_valid_return = actual_valid[return_target].clip(-0.12, 0.15)
    train_reg = lgb.Dataset(
        actual_train[features],
        label=clipped_train_return,
        weight=actual_train['sample_weight']
    )
    valid_reg = lgb.Dataset(
        actual_valid[features],
        label=clipped_valid_return,
        reference=train_reg
    )
    reg_params = dict(common_params)
    reg_params.update({'objective': 'regression_l1', 'metric': 'l1'})
    reg_model = lgb.train(
        reg_params,
        train_reg,
        num_boost_round=900,
        valid_sets=[valid_reg],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )

    valid_scored = actual_valid.copy()
    valid_scored['win_probability'] = cls_model.predict(valid_scored[features], num_iteration=cls_model.best_iteration)
    valid_scored['predicted_net_return'] = reg_model.predict(valid_scored[features], num_iteration=reg_model.best_iteration)
    valid_scored['expected_profit_score'] = (
        valid_scored['win_probability'] * valid_scored['predicted_net_return'].clip(lower=0.0, upper=0.10)
    )
    
    # 严苛级漏斗过滤
    valid_scored_filtered = _pool_rule_filter(valid_scored, pool_type, relax_level=0, verbose=True)
    threshold, stats = _select_validation_threshold(valid_scored_filtered, return_target, max_positions, allow_negative_expect=False)

    test_data = df[df['日期'] == predict_date].copy()
    test_data[features] = test_data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_data['win_probability'] = cls_model.predict(test_data[features], num_iteration=cls_model.best_iteration)
    test_data['predicted_net_return'] = reg_model.predict(test_data[features], num_iteration=reg_model.best_iteration)
    test_data['expected_profit_score'] = (
        test_data['win_probability'] * test_data['predicted_net_return'].clip(lower=0.0, upper=0.10)
    )

    test_pool = pd.DataFrame()
    if np.isfinite(threshold):
        test_pool_raw = _pool_rule_filter(test_data, pool_type, relax_level=0, verbose=False)
        test_pool = test_pool_raw[
            (test_pool_raw['expected_profit_score'] >= threshold) &
            (test_pool_raw['predicted_net_return'] > 0.002)
        ].copy()

    # 🚨 比赛解冻机制：若常规严苛判定无果，系统逐级降阻
    if test_pool.empty and COMPETITION_URGENCY_MODE:
        print(f"🔄 [比赛解冻] {pool_cn} 无标的通过，启动三级自适应释放漏斗...")
        for r_lvl in [1, 2]:
            print(f"   ➡️ 尝试解冻 Level {r_lvl} 选股过滤矩阵...")
            valid_scored_relax = _pool_rule_filter(valid_scored, pool_type, relax_level=r_lvl, verbose=True)
            r_threshold, r_stats = _select_validation_threshold(
                valid_scored_relax, return_target, max_positions, allow_negative_expect=True
            )
            
            if np.isfinite(r_threshold):
                test_pool_raw = _pool_rule_filter(test_data, pool_type, relax_level=r_lvl, verbose=False)
                test_pool = test_pool_raw[
                    (test_pool_raw['expected_profit_score'] >= r_threshold) &
                    (test_pool_raw['predicted_net_return'] > -0.002)
                ].copy()
                
                if not test_pool.empty:
                    print(f"   🎉 [解冻成功] 采用 Level {r_lvl} 松弛漏斗，成功复活 {len(test_pool)} 只优选标的！")
                    threshold = r_threshold
                    stats = r_stats
                    break
            else:
                print(f"   ❌ Level {r_lvl} 漏斗回测，模型仍判定无正贡献。")

    # 🚨 比赛终极交易复活舱 (绝对不空仓)
    if test_pool.empty and COMPETITION_URGENCY_MODE:
        print(f"🔥 [终极强制复活舱] 已连续多日无股可买！强行在 Level 2 安全过滤池中，直接提取 AI 综合评分前 {max_positions} 的标的！")
        test_pool_raw = _pool_rule_filter(test_data, pool_type, relax_level=2, verbose=False)
        if not test_pool_raw.empty:
            test_pool = test_pool_raw.sort_values('expected_profit_score', ascending=False).head(max_positions)
            stats = {'avg_return': 0.001, 'win_rate': 0.40, 'trades': len(test_pool)}
            test_pool['验证集平均收益'] = stats['avg_return']
            test_pool['池类型'] = pool_cn
            return test_pool

    if test_pool.empty or not np.isfinite(threshold):
        print(f"⛔ {pool_cn} 今日无安全机会，策略强制空仓防守。")
        return pd.DataFrame()

    print(f"✅ {pool_cn} 选择：{stats['trades']} 笔交易 | 回测均值 {stats['avg_return']*100:.2f}% | 胜率 {stats['win_rate']*100:.1f}%")
    test_pool['验证集平均收益'] = stats['avg_return']
    test_pool['池类型'] = pool_cn

    cls_model.save_model(os.path.join(MODEL_DIR, f'lgbm_{pool_type}_classifier.txt'))
    reg_model.save_model(os.path.join(MODEL_DIR, f'lgbm_{pool_type}_return.txt'))
    return test_pool.sort_values('expected_profit_score', ascending=False).head(max_positions * 4)


def _allocate_limited_capital(hot_candidates, steady_candidates):
    candidates = pd.concat([hot_candidates, steady_candidates], ignore_index=True, sort=False)
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidates = candidates.sort_values('expected_profit_score', ascending=False)
    selected_rows = []
    hot_count = 0
    used_codes = set()
    
    # 强制最多分配 2 只（极端高确定性下可拓展到 3 只），杜绝小金额起征点惩罚
    dynamic_max_positions = MAX_TOTAL_POSITIONS
    if TOTAL_CAPITAL < 35000:
        dynamic_max_positions = 2  
        
    for _, row in candidates.iterrows():
        code = str(row['股票代码']).zfill(6)
        if code in used_codes:
            continue
        if row['池类型'] == '突击池' and hot_count >= MAX_HOT_POSITIONS:
            continue
        selected_rows.append(row.copy())
        used_codes.add(code)
        if row['池类型'] == '突击池':
            hot_count += 1
        if len(selected_rows) >= dynamic_max_positions:
            break

    if not selected_rows:
        return pd.DataFrame(), pd.DataFrame()

    selected = pd.DataFrame(selected_rows)
    selected['建议挂单限价'] = np.where(
        selected['池类型'] == '突击池',
        selected['收盘'] * HOT_LIMIT_DISCOUNT,
        selected['收盘'] * STEADY_LIMIT_DISCOUNT
    ).round(2)
    selected['防守参考价'] = np.where(
        selected['池类型'] == '突击池',
        selected['建议挂单限价'] * 0.955,
        selected['建议挂单限价'] * 0.96
    ).round(2)
    selected['计划退出'] = '成交后持有至 T+2 收盘'
    selected['买入股数'] = 0
    selected['实际消耗资金'] = 0.0

    remaining_cash = float(TOTAL_CAPITAL)
    position_count = len(selected)
    base_cash = remaining_cash / max(position_count, 1)

    valid_indices = []
    for idx, row in selected.iterrows():
        price = float(row['建议挂单限价'])
        target_cash = min(base_cash, remaining_cash)
        
        if target_cash < MIN_POSITION_CASH:
            continue
            
        shares = int(target_cash // (price * 100)) * 100
        while shares > 0:
            buy_amount = shares * price
            buy_fee = max(buy_amount * BUY_COMMISSION_RATE, MIN_COMMISSION)
            total_cost = buy_amount + buy_fee
            if total_cost <= remaining_cash:
                break
            shares -= 100
            
        if shares <= 0:
            continue

        buy_amount = shares * price
        buy_fee = max(buy_amount * BUY_COMMISSION_RATE, MIN_COMMISSION)
        total_cost = buy_amount + buy_fee
        
        selected.at[idx, '买入股数'] = int(shares)
        selected.at[idx, '实际消耗资金'] = round(total_cost, 2)
        selected.at[idx, '买入理由'] = (
            f"AI期望: {row['expected_profit_score']:.4f} | "
            f"预测净盈: {row['predicted_net_return']*100:.2f}% | "
            f"往返起免比: {buy_fee/buy_amount*100:.2f}%"
        )
        remaining_cash -= total_cost
        valid_indices.append(idx)

    selected = selected.loc[valid_indices].copy()
    final_hot = selected[selected['池类型'] == '突击池'].copy()
    final_steady = selected[selected['池类型'] == '稳健池'].copy()
    return final_hot, final_steady


def train_and_predict(df, target_trade_date=None):
    print("\n🧠 [AI智能中心] 启动双机器学习模型极速求解中...")

    features = [
        '涨跌幅', '振幅', '换手率', 'Return_1d', 'Gap_Open',
        'Close_Open_Ratio', 'High_Low_Ratio', 'Close_High_Ratio', 'Close_Low_Ratio',
        'Upper_Shadow', 'Lower_Shadow', 'MA5_bias', 'MA10_bias', 'MA20_bias', 'MA60_bias',
        'MA5_MA20', 'MA20_MA60', 'Momentum_3d', 'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
        'Volatility_10d', 'Volatility_20d', 'ATR_14', 'Price_Position_20', 'Up_Days_5',
        'Vol_Ratio_5', 'Vol_Ratio_20', 'Turnover_MA5', 'Turnover_MA20', 'Turnover_Accel',
        'Amount_Log', 'Illiquidity_10', 'Price_Vol_Corr_10', 'RSI_14',
        '换手率截面排名', '成交额截面排名', '动量5日截面排名', '动量20日截面排名',
        '20日位置截面排名', '波动率截面排名', '市值代理排名',
        '市场上涨比例', '市场中位涨幅', '市场平均涨幅', '市场波动中位数',
        'Vol_Spike_5d', 'Intraday_Ret_Ratio', 'Limit_Up_Proximity', 'Drop_From_High'
    ]

    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"❌ 数据集缺少特征列: {missing}")
        return

    df = df.copy()
    df['日期'] = df['日期'].astype(str)
    # predict_date 是模型使用的已收盘特征日，不是实际下单日。
    predict_date = df['日期'].max()
    target_trade_date = target_trade_date or get_next_trade_date(predict_date)

    hot_candidates = _fit_pool_models(
        df, features, predict_date, 'hot',
        'target_hot_win', 'target_hot_return', MAX_HOT_POSITIONS
    )
    steady_candidates = _fit_pool_models(
        df, features, predict_date, 'steady',
        'target_steady_win', 'target_steady_return', MAX_TOTAL_POSITIONS
    )

    final_hot, final_steady = _allocate_limited_capital(hot_candidates, steady_candidates)
    _print_trading_dashboard(final_hot, final_steady, predict_date, target_trade_date)
    record_predictions_to_ledger(final_hot, final_steady, predict_date, target_trade_date)


def record_predictions_to_ledger(final_hot, final_steady, predict_date, target_trade_date=None):
    target_trade_date = target_trade_date or get_next_trade_date(predict_date)
    new_rows = []
    for pool_df in [final_hot, final_steady]:
        for _, row in pool_df.iterrows():
            new_rows.append({
                # 预测日期保留为“信号基准日”，供既有 T+1/T+2 对账逻辑使用。
                '预测日期': predict_date,
                '目标交易日': target_trade_date,
                '代码': str(row['股票代码']).zfill(6),
                '名称': row['股票名称'],
                '池类型': row['池类型'],
                'AI得分': round(float(row['win_probability']), 6),
                '预测净收益率': round(float(row['predicted_net_return']), 6),
                '计划挂单买入价': float(row['建议挂单限价']),
                '买入股数': int(row['买入股数']),
                '计划占用资金': float(row['实际消耗资金']),
                '实际T1买入价格': np.nan,
                '实际T2收盘': np.nan,
                '买入费用': np.nan,
                '卖出费用': np.nan,
                '真实盈亏': np.nan,
                '真实收益率': np.nan,
                '结账状态': '未结账'
            })

    if not new_rows:
        print("📝 本交易日无通过验证与自适应松弛阈值的优选标的。")
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(LEDGER_PATH):
        df_old = pd.read_csv(LEDGER_PATH, dtype={'代码': str})
        df_combined = pd.concat([df_old, df_new], ignore_index=True, sort=False)
        df_combined = df_combined.drop_duplicates(subset=['预测日期', '代码'], keep='first')
    else:
        df_combined = df_new
    df_combined.to_csv(LEDGER_PATH, index=False, encoding='utf-8-sig')
    print(f"📝 {target_trade_date} 的 {len(new_rows)} 只自适应优选标的已成功落盘记账。")


def reconcile_ledger_performance():
    if not os.path.exists(LEDGER_PATH):
        return
    df_ledger = pd.read_csv(LEDGER_PATH, dtype={'代码': str})
    if df_ledger.empty:
        return

    required_defaults = {
        '买入股数': 100, '计划占用资金': np.nan, '买入费用': np.nan,
        '卖出费用': np.nan, '真实盈亏': np.nan, '预测净收益率': np.nan
    }
    for col, default in required_defaults.items():
        if col not in df_ledger.columns:
            df_ledger[col] = default

    unsettled_mask = df_ledger['结账状态'] == '未结账'
    if not unsettled_mask.any():
        _print_performance_report(df_ledger)
        return

    print("\n" + "🔄" * 12 + " 破除生存漏洞·实盘对账引擎 " + "🔄" * 12)
    updated_count = 0

    for idx, row in df_ledger[unsettled_mask].iterrows():
        code = str(row['代码']).split('.')[0].zfill(6)
        pred_date_str = str(row['预测日期'])[:10]
        file_path = os.path.join(HIST_DATA_DIR, f"{code}.csv")
        if not os.path.exists(file_path):
            continue

        try:
            df_hist = pd.read_csv(file_path)
            df_hist = df_hist.sort_values('日期').reset_index(drop=True)
            df_hist['日期'] = df_hist['日期'].astype(str)
            future_dates = df_hist[df_hist['日期'] > pred_date_str]
            if len(future_dates) < 2:
                continue

            t1_row = future_dates.iloc[0]
            
            limit_buy_price = float(row['计划挂单买入价'])
            shares_value = pd.to_numeric(row.get('买入股数', 100), errors='coerce')
            shares = int(shares_value) if pd.notna(shares_value) and shares_value > 0 else 100

            t1_open = float(t1_row['开盘'])
            t1_low = float(t1_row['最低'])
            t1_high = float(t1_row['最高'])
            t1_close = float(t1_row['收盘'])
            t1_preclose = float(t1_row.get('preclose', t1_open / (1 + float(t1_row['涨跌幅'])/100.0)))
            
            is_limit_down_t1 = (t1_high == t1_low) and (t1_close <= t1_preclose * 0.91)
            is_limit_up_t1 = (t1_high == t1_low) and (t1_close >= t1_preclose * 1.09)
            
            can_buy = (t1_low <= limit_buy_price) and not is_limit_down_t1 and not is_limit_up_t1
            
            if can_buy:
                actual_buy_price = min(limit_buy_price, t1_open)
                actual_buy_price = max(actual_buy_price, t1_low) 

                sell_idx = 1
                t2_row = future_dates.iloc[sell_idx]
                
                while sell_idx < len(future_dates):
                    t_sell_row = future_dates.iloc[sell_idx]
                    t_sell_high = float(t_sell_row['最高'])
                    t_sell_low = float(t_sell_row['最低'])
                    t_sell_close = float(t_sell_row['收盘'])
                    t_sell_open = float(t_sell_row['开盘'])
                    t_sell_pre = float(t_sell_row.get('preclose', t_sell_open / (1 + float(t_sell_row['涨跌幅'])/100.0)))
                    
                    is_limit_down_curr = (t_sell_high == t_sell_low) and (t_sell_close <= t_sell_pre * 0.91)
                    
                    if not is_limit_down_curr:
                        t2_row = t_sell_row
                        break
                    else:
                        sell_idx += 1  
                
                if sell_idx >= len(future_dates):
                    continue

                t2_close = float(t2_row['收盘'])
                buy_amount = actual_buy_price * shares
                sell_amount = t2_close * shares
                
                buy_fee = max(buy_amount * BUY_COMMISSION_RATE, MIN_COMMISSION)
                sell_fee = max(sell_amount * SELL_COMMISSION_RATE, MIN_COMMISSION) + sell_amount * STAMP_DUTY_RATE
                
                pnl = sell_amount - sell_fee - buy_amount - buy_fee
                invested = buy_amount + buy_fee
                real_return = pnl / invested if invested > 0 else 0.0

                df_ledger.at[idx, '实际T1买入价格'] = round(actual_buy_price, 2)
                df_ledger.at[idx, '实际T2收盘'] = round(t2_close, 2)
                df_ledger.at[idx, '买入费用'] = round(buy_fee, 2)
                df_ledger.at[idx, '卖出费用'] = round(sell_fee, 2)
                df_ledger.at[idx, '真实盈亏'] = round(pnl, 2)
                df_ledger.at[idx, '真实收益率'] = round(real_return, 6)
                df_ledger.at[idx, '计划占用资金'] = round(invested, 2)
                df_ledger.at[idx, '结账状态'] = '已结账' if sell_idx == 1 else f'已结账(延至T+{sell_idx+1})'
            else:
                df_ledger.at[idx, '结账状态'] = '未成交失效'
            updated_count += 1
        except Exception as e:
            print(f"⚠️ 对账失败 {code}: {e}")

    if updated_count > 0:
        df_ledger.to_csv(LEDGER_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 对账完成：基于生存滑点和跌停延期策略更新 {updated_count} 笔。")
    _print_performance_report(df_ledger)


def _print_performance_report(df_ledger):
    settled = df_ledger[df_ledger['结账状态'].str.startswith('已结账', na=False)].copy()
    total_orders = len(df_ledger[df_ledger['结账状态'].isin(['已结账', '未成交失效'])])
    if settled.empty:
        print("📊 [业绩看板] 暂无已结账平仓记录。")
        return

    settled['真实收益率'] = pd.to_numeric(settled['真实收益率'], errors='coerce').fillna(0.0)
    settled['真实盈亏'] = pd.to_numeric(settled['真实盈亏'], errors='coerce').fillna(0.0)
    settled['计划占用资金'] = pd.to_numeric(settled['计划占用资金'], errors='coerce').fillna(0.0)

    wins = settled[settled['真实盈亏'] > 0]
    total_pnl = settled['真实盈亏'].sum()
    total_invested = settled['计划占用资金'].sum()
    win_rate = len(wins) / len(settled)
    fill_rate = len(settled) / total_orders if total_orders else 0.0
    capital_return = total_pnl / TOTAL_CAPITAL
    weighted_trade_return = total_pnl / total_invested if total_invested > 0 else 0.0

    daily_pnl = settled.groupby('预测日期')['真实盈亏'].sum()
    best_day = daily_pnl.max() if not daily_pnl.empty else 0.0
    worst_day = daily_pnl.min() if not daily_pnl.empty else 0.0

    print("\n" + "🏆" * 16 + " 比赛实盘对账业绩看板 " + "🏆" * 16)
    print(f"📊 追踪区间：{settled['预测日期'].min()}  >>>  {settled['预测日期'].max()}")
    print(f"🎯 已平仓：{len(settled)} 笔 | 成交率：{fill_rate*100:.1f}% | 真实胜率：{win_rate*100:.1f}%")
    print(f"💰 累计总盈亏：{total_pnl:.2f} 元 | 比赛整体收益率：{capital_return*100:.2f}%")
    print(f"📈 交易加权胜出率：{weighted_trade_return*100:.2f}% (真实扣除往返佣金惩罚后)")
    print(f"🌞 单日最牛斩获：{best_day:.2f} 元 | 🌧️ 单日最惨回撤：{worst_day:.2f} 元")
    print("🏆" * 45 + "\n")


def _print_trading_dashboard(final_hot, final_steady, predict_date, target_trade_date=None):
    target_trade_date = target_trade_date or get_next_trade_date(predict_date)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 260)

    def format_df(df_format):
        if df_format.empty:
            return df_format
        df_f = df_format.copy()
        df_f['模型胜率分'] = (df_f['win_probability'] * 100).round(1).astype(str) + '%'
        df_f['预测净收益'] = (df_f['predicted_net_return'] * 100).round(2).astype(str) + '%'
        df_f['建议买入'] = df_f['买入股数'].astype(int).astype(str) + ' 股'
        df_f['计划'] = df_f['计划退出'] + ' | 防守参考:' + df_f['防守参考价'].astype(str)
        return df_f[[
            '股票代码', '股票名称', '收盘', '建议挂单限价', '建议买入',
            '实际消耗资金', '模型胜率分', '预测净收益', '计划', '买入理由'
        ]]

    total_used = 0.0
    for part in [final_hot, final_steady]:
        if not part.empty:
            total_used += part['实际消耗资金'].sum()

    print("\n\n" + "★" * 145)
    print(" " * 42 + f"🚀 自适应极速抢单量化指令单 - {target_trade_date} 🚀")
    print("★" * 145)
    print(f"数据基准日：{predict_date} | 预测/下单目标交易日：{target_trade_date}")
    print(f"计划占用总资金：{total_used:.2f} / {TOTAL_CAPITAL:.2f} 元 | 预留空闲现金：{TOTAL_CAPITAL-total_used:.2f} 元")
    print("\n⚔️ 突击池（高能脉冲，防一字板洗盘）\n" + "-" * 145)
    print(format_df(final_hot).to_string(index=False) if not final_hot.empty else "今日无通过自适应松弛阈值的突击标的。")
    print("\n🛡️ 稳健池（回撤超跌，低吸防守反击）\n" + "-" * 145)
    print(format_df(final_steady).to_string(index=False) if not final_steady.empty else "今日无通过自适应松弛阈值的稳健标的。")
    print("=" * 145 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='A 股量化分析：自动补齐漏跑交易日的行情、新闻和情绪特征。'
    )
    parser.add_argument('--backfill-start', help='手工补采开始日期，格式 YYYY-MM-DD')
    parser.add_argument('--backfill-end', help='手工补采结束日期，格式 YYYY-MM-DD；默认到最近交易日')
    parser.add_argument(
        '--force-news', action='store_true',
        help='重抓指定日期已有的新闻及 LLM 特征；不指定日期时仅重抓最近交易日'
    )
    parser.add_argument(
        '--backfill-only', action='store_true',
        help='只执行缺口补采，不训练模型、不生成交易指令'
    )
    return parser.parse_args()


def collect_news_for_date(target_date, force=False, spot_data=None):
    """幂等完成某个交易日的行情快照、原始新闻和 LLM 特征。"""
    target_str = target_date.replace('-', '')
    news_file = os.path.join(ALT_DATA_DIR, f'news_{target_str}.csv')

    if os.path.exists(news_file) and not force:
        news_data = pd.read_csv(news_file, dtype={'关联代码': str})
        print(f"📦 {target_date} 原始新闻已归档，跳过网络重抓。")
    else:
        if spot_data is None:
            spot_data = update_daily_price_with_baostock(target_date, target_str)
        news_data = fetch_alternative_data(
            spot_data, target_date, target_str, force=force
        ) if spot_data is not None else None

    llm_features = batch_process_news(news_data, target_str, force=force)
    return spot_data, news_data, llm_features


def main():
    start_time = time.time()
    init_environment()
    args = parse_args()
    
    is_macro_risk = False
    latest_str, latest_date = get_latest_trade_date()
    target_trade_date = get_next_trade_date(latest_date)
    print(f"📅 [系统就绪] 最新可用数据基准日：{latest_date}")
    print(f"🎯 [预测目标] 下一交易日（下单日）：{target_trade_date}")

    try:
        pending_dates = discover_backfill_dates(
            latest_date,
            start_date=args.backfill_start,
            end_date=args.backfill_end,
            force=args.force_news
        )
    except (ValueError, TypeError) as e:
        raise SystemExit(f"❌ 补采日期参数错误: {e}")

    latest_spot_data = None
    if pending_dates:
        print(f"🔁 待补采交易日：{', '.join(pending_dates)}")
        for target_date in pending_dates:
            supplied_spot = latest_spot_data if target_date == latest_date else None
            spot, _, _ = collect_news_for_date(
                target_date, force=args.force_news, spot_data=supplied_spot
            )
            if target_date == latest_date:
                latest_spot_data = spot
    else:
        print("✅ 新闻归档与情绪特征没有发现缺口。")

    if args.backfill_only:
        print(f"\n⏱️ 补采任务结束。总计耗时: {time.time() - start_time:.1f} 秒。")
        return
        
    market_breadth = calculate_market_breadth(latest_date)
        
    if market_breadth < 0.30:
        is_macro_risk = True
        print("🚨🚨 警告：今日全市场处于极速绞杀市（上涨家数不足30%）！触发铁血熔断空仓保护！")
    elif market_breadth < 0.35:
        print("⚠️ 警告：大盘风向偏弱（广度 30%-35%），进入【防守反击】模式，将自动收缩仓位集中力量攻坚龙头。")
            
    if latest_spot_data is None:
        latest_spot_data = update_daily_price_with_baostock(latest_date, latest_str)

    print("🌐 接入 NLP 情绪过滤器（加载全部历史归档）...")
    df_llm_cleaned = load_all_llm_features()
        
    df_dataset = build_machine_learning_dataset(df_llm_cleaned, latest_str)
    
    if df_dataset is not None and not df_dataset.empty: 
        if is_macro_risk:
            print(f"\n🚨 【极度危险】：大盘崩盘期，{target_trade_date} 策略强制静默空仓。")
        else:
            train_and_predict(df_dataset, target_trade_date)
    else:
        print("\n❌ 致命错误：数据加载或更新失败。")
        
    reconcile_ledger_performance()
    print(f"\n⏱️ 策略运行完毕。总计耗时: {time.time() - start_time:.1f} 秒。")


if __name__ == "__main__":
    main()
