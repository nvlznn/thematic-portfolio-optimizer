import os
import pandas as pd

def clean_daily_prices():
    print("[Module: clean.py] 啟動日頻率股價精準清洗流程...")
    raw_path = "data/raw/202105"
    processed_path = "data/processed/202105"
    os.makedirs(processed_path, exist_ok=True)
    
    file_path = os.path.join(raw_path, "raw_prices.csv")
    
    try:
        df = pd.read_csv(file_path, encoding="utf-16", sep="\t", quoting=3)
    except Exception:
        df = pd.read_csv(file_path, encoding="cp950", sep=",", quoting=3)
        
    print(f"原始價格表成功載入，共計 {len(df)} 行。")
    
    df.rename(columns={
        '證券代碼': 'stock_id', 
        '年月日': 'date',
        '收盤價(元)': 'close',
        '成交量(千股)': 'volume',
        '成交值(千元)': 'amount'
    }, inplace=True)
    
    # 切分：使用 r'\s+' 解決所有空白
    df['stock_id'] = df['stock_id'].astype(str).str.strip()
    df['stock_id'] = df['stock_id'].str.split(r'\s+').str[0]
    
    # 解析日期 (datetime64[ns])
    df['date'] = pd.to_datetime(df['date'].astype(str).str.split('.').str[0], format="%Y%m%d", errors='coerce')
    
    df = df[df['stock_id'] != 'Y9999'].copy()
    df.dropna(subset=['date'], inplace=True)
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce') * 1000
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce') * 1000
    
    # 第一階段排序：為了停牌補值的連續性
    df.sort_values(by=['stock_id', 'date'], inplace=True)
    df['is_suspended'] = df['close'].isna()
    df['close'] = df.groupby('stock_id')['close'].ffill(limit=3)
    
    df.loc[df['is_suspended'] & df['close'].notna(), 'volume'] = 0.0
    df.loc[df['is_suspended'] & df['close'].notna(), 'amount'] = 0.0
    
    df.dropna(subset=['close'], inplace=True)
    
    df['volume'] = df['volume'].astype('int64')
    df['amount'] = df['amount'].astype('float64')
    
    # 第二階段排序：為了配合下游回測引擎的效能(先排日期，再排代碼)
    df.sort_values(by=['date', 'stock_id'], inplace=True)
    
    # 嚴格篩選欄位與輸出
    output_df = df[['date', 'stock_id', 'close', 'volume', 'amount']]
    output_df.to_parquet(os.path.join(processed_path, "prices.parquet"), index=False)
    
    print(f"prices.parquet 建立成功！最終有效回測資料：{len(output_df)} 行。")

if __name__ == "__main__":
    clean_daily_prices()