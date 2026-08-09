import akshare as ak
import baostock as bs
import pandas as pd
import os
import time
import socket
from datetime import datetime
import sys
# 修复普通 print 的输出编码
sys.stdout.reconfigure(encoding='utf-8')

# 修复 tqdm 等进度条（标准错误流）的输出编码
sys.stderr.reconfigure(encoding='utf-8')

# ================= 核心防御区 =================
# 🔥 注入强心针：强制设定底层网络超时时间为 15 秒！
# 一旦 BaoStock 服务器不返回数据超过 15 秒，立刻抛出异常，绝不卡死！
socket.setdefaulttimeout(15)

# ================= 配置区 =================
START_DATE = "2021-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
SAVE_DIR = "./stock_data_hist/historical_daily"
# ==========================================

def get_stock_list():
    """使用 AkShare 获取股票列表（这个接口不封云服务器）"""
    print("🔄 正在获取基础股票列表...")
    df_info = ak.stock_info_a_code_name()
    df_info = df_info[~df_info['name'].str.contains('ST|退|退市')]
    df_info = df_info[df_info['code'].str.startswith(('60', '00'))]
    return dict(zip(df_info['code'], df_info['name']))

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    stock_dict = get_stock_list()
    total_stocks = len(stock_dict)
    print(f"✅ 列表就绪！主板股票共计: {total_stocks} 只。\n")

    print("🚀 启动 BaoStock 引擎...")
    bs.login()

    success_count = 0
    actual_request = 0

    for i, (stock_code, stock_name) in enumerate(stock_dict.items(), 1):
        file_path = os.path.join(SAVE_DIR, f"{stock_code}.csv")

        # 1. 断点续传
        if os.path.exists(file_path):
            print(f"[{i}/{total_stocks}] ⏭️ {stock_name}({stock_code}) 文件已存在，跳过。")
            success_count += 1
            continue

        # 2. 定期清理内存与连接 (降低到每 50 只重置一次)
        actual_request += 1
        if actual_request % 50 == 0:
            print(f"\n🔄 [系统调度] 防止服务端挂起，主动重启 BaoStock 连接...")
            bs.logout()
            time.sleep(1)
            bs.login()

        try:
            bs_code = f"sh.{stock_code}" if stock_code.startswith('60') else f"sz.{stock_code}"
            
            # 这里如果卡住，15秒后会触发 Timeout 异常
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="d",
                adjustflag="2"  # 前复权
            )

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                print(f"[{i}/{total_stocks}] ⚠️ {stock_name}({stock_code}) 无交易数据。")
                continue

            # 3. 数据清洗与组装
            df = pd.DataFrame(data_list, columns=rs.fields)
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
            df['涨跌额'] = (df['close'] - df['preclose']).round(2)

            df_final = pd.DataFrame({
                '日期': df['date'],
                '股票代码': stock_code,
                '股票名称': stock_name,
                '开盘': df['open'].round(2),
                '收盘': df['close'].round(2),
                '最高': df['high'].round(2),
                '最低': df['low'].round(2),
                '成交量': df['volume'],
                '成交额': df['amount'],
                '振幅': df['振幅'],
                '涨跌幅': df['pctChg'].round(2),
                '涨跌额': df['涨跌额'],
                '换手率': df['turn'].round(4)
            })

            df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"[{i}/{total_stocks}] ✅ {stock_name}({stock_code}) 下载成功 ({len(df_final)}天)。")
            success_count += 1
            
            time.sleep(0.1)

        # 🔥 核心急救区：如果发生网络断开或 15 秒超时卡死，在这里强制抢救
        except Exception as e:
            print(f"[{i}/{total_stocks}] ❌ {stock_name}({stock_code}) 触发异常/超时: {str(e)}")
            print("🏥 [系统抢救] 检测到连接假死，正在强制重启 BaoStock...")
            try:
                bs.logout()
            except:
                pass
            time.sleep(2)
            bs.login()  # 重新登录，下一只股票继续正常下载

    bs.logout()
    print("\n" + "=" * 40)
    print(f"🎉 物理隔离穿透成功！成功落盘: {success_count} / {total_stocks}")

if __name__ == "__main__":
    main()
