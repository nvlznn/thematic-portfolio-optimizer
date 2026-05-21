import os
import pandas as pd

def generate_candidate_pool():
    print("[Module: candidates.py] 啟動候選股池精準建立流程...")
    raw_path = "data/raw/202105"
    processed_path = "data/processed/202105"
    os.makedirs(processed_path, exist_ok=True)
    
    # 1. 讀取市值原始表
    cap_path = os.path.join(raw_path, "raw_market_cap.csv")
    try:
        df_cap = pd.read_csv(cap_path, encoding="utf-16", sep="\t", quoting=3)
    except Exception:
        df_cap = pd.read_csv(cap_path, encoding="cp950", sep=",", quoting=3)
        
    df_cap.rename(columns={'證券代碼': 'stock_id', '年月日': 'date', '市值(百萬元)': 'market_cap_raw'}, inplace=True)
    
    # 市值表代碼切分：使用 r'\s+'
    df_cap['stock_id'] = df_cap['stock_id'].astype(str).str.strip()
    df_cap['stock_id'] = df_cap['stock_id'].str.split(r'\s+').str[0]
    
    df_cap['date'] = pd.to_datetime(df_cap['date'].astype(str).str.split('.').str[0], format="%Y%m%d", errors='coerce')
    df_cap['market_cap'] = pd.to_numeric(df_cap['market_cap_raw'], errors='coerce') * 1000000
    
    # 2. 讀取價格原始表
    prices_raw_path = os.path.join(raw_path, "raw_prices.csv")
    try:
        df_orig = pd.read_csv(prices_raw_path, encoding="utf-16", sep="\t", quoting=3)
    except Exception:
        df_orig = pd.read_csv(prices_raw_path, encoding="cp950", sep=",", quoting=3)
        
    df_orig.rename(columns={'證券代碼': 'raw_id'}, inplace=True)
    df_orig['raw_id'] = df_orig['raw_id'].astype(str).str.strip()
    
    df_meta_raw = df_orig.groupby('raw_id').last().reset_index()
    
    # 核心切分：用 r'\s+' 切分，n=1 代表只切第一刀
    # 這樣會產生兩個欄位：[0]是代碼，[1]是後面的所有名字
    split_df = df_meta_raw['raw_id'].str.split(r'\s+', n=1, expand=True)
    
    df_meta_raw['stock_id'] = split_df[0]
    # 如果有名字就填名字，萬一只有代碼（沒有空白），就拿代碼補上防呆
    df_meta_raw['name'] = split_df[1].fillna(split_df[0]) 
    
    df_meta_raw['industry'] = df_meta_raw['TSE產業_名稱'].astype(str).str.strip()
    
    df_meta = df_meta_raw[['stock_id', 'name', 'industry']]
    
    # 3. 鎖定最新調倉基準日的市值
    latest_date = df_cap['date'].max()
    df_latest_cap = df_cap[df_cap['date'] == latest_date][['stock_id', 'market_cap']]
    
    # 內連結合併
    df_candidates = pd.merge(df_meta, df_latest_cap, on='stock_id', how='inner')
    df_candidates['theme_tag'] = "high_div"
    
    # 轉型態對齊契約
    df_candidates['stock_id'] = df_candidates['stock_id'].astype(str)
    df_candidates['name'] = df_candidates['name'].astype(str)
    df_candidates['industry'] = df_candidates['industry'].astype(str)
    df_candidates['market_cap'] = df_candidates['market_cap'].astype('float64')
    df_candidates['theme_tag'] = df_candidates['theme_tag'].astype(str)
    
    output_df = df_candidates[['stock_id', 'name', 'market_cap', 'industry', 'theme_tag']].dropna()
    output_df.to_parquet(os.path.join(processed_path, "candidates.parquet"), index=False)
    
    print(f"candidates.parquet 導出成功！共有 {len(output_df)} 檔純淨的個股候選標的。")

if __name__ == "__main__":
    generate_candidate_pool()