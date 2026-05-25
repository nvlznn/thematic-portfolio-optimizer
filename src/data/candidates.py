import os
import pandas as pd

def generate_candidate_pool():
    print("[Module: candidates.py] 啟動候選股池精準建立流程...")
    raw_path = "data/raw/202105"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)
    
    # 1. 讀取市值原始表
    cap_path = os.path.join(raw_path, "raw_market_cap.csv")
    try:
        df_cap = pd.read_csv(cap_path, encoding="utf-16", sep="\t", quoting=3)
    except Exception:
        df_cap = pd.read_csv(cap_path, encoding="cp950", sep=",", quoting=3)
        
    df_cap.rename(columns={'證券代碼': 'stock_id', '年月日': 'date', '市值(百萬元)': 'market_cap_raw'}, inplace=True)
    
    # 市值表代碼切分
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
    
    # 核心切分
    split_df = df_meta_raw['raw_id'].str.split(r'\s+', n=1, expand=True)
    df_meta_raw['stock_id'] = split_df[0]
    df_meta_raw['name'] = split_df[1].fillna(split_df[0]) 
    df_meta_raw['industry'] = df_meta_raw['TSE產業_名稱'].astype(str).str.strip()
    
    df_meta = df_meta_raw[['stock_id', 'name', 'industry']]
    
    # 3. 鎖定最新調倉基準日的市值
    latest_date = df_cap['date'].max()
    df_latest_cap = df_cap[df_cap['date'] == latest_date][['stock_id', 'market_cap']]
    
    # 內連結合併
    df_candidates = pd.merge(df_meta, df_latest_cap, on='stock_id', how='inner')
    
    # ==========================================
    # ✨ 核心新增：根據產業別動態貼上主題標籤
    # ==========================================
    # 你可以跟小組討論後，自由增減這個字典裡面的分類邏輯
    theme_mapping = {
        # 1. 科技硬體與軟體 (Technology)
        "半導體業": "tech",
        "光電業": "tech",
        "其他電子業": "tech",
        "通訊網路業": "tech",
        "資訊服務業": "tech",
        "電子工業": "tech",
        "電子通路業": "tech",
        "電子零組件": "tech",
        "電腦及週邊設備業": "tech",
        "數位雲端": "tech",
        
        # 2. 原物料與基礎工業 (Materials) - 景氣高度敏感
        "水泥工業": "materials",
        "玻璃陶瓷": "materials",
        "造紙工業": "materials",
        "塑膠工業": "materials",
        "橡膠工業": "materials",
        "鋼鐵工業": "materials",
        "化學工業": "materials",
        
        # 3. 工業製造與設備 (Industrials) - 資本支出相關
        "電機機械": "industrials",
        "電器電纜": "industrials",
        
        # 4. 非必需消費/景氣循環 (Consumer Discretionary) - 經濟好才消費
        "汽車工業": "consumer_disc",
        "貿易百貨": "consumer_disc",
        "運動休閒": "consumer_disc",
        "文化創意業": "consumer_disc",
        "居家生活": "consumer_disc",
        "觀光餐旅": "consumer_disc",
        
        # 5. 必需消費/防禦型 (Consumer Staples) - 經濟差也要消費
        "食品工業": "consumer_staples",
        
        # 6. 金融 (Financials) - 升降息敏感
        "金融業": "finance",
        
        # 7. 不動產 (Real Estate)
        "建材營造": "real_estate",
        
        # 8. 交通運輸 (Transportation)
        "航運業": "transportation",
        
        # 9. 醫療保健 (Healthcare)
        "生技醫療業": "healthcare",
        "農業科技": "healthcare",
        
        # 10. 公用事業與能源 (Energy & Utilities)
        "油電燃氣業": "energy_utility",
        "綠能環保": "energy_utility",
        
        # 11. 其他/無法歸類 (General)
        "存託憑證": "general",
        "其他": "general",
        "": "general" 
    }
    
    df_candidates['theme_tag'] = df_candidates['industry'].map(theme_mapping).fillna("general")
    # ==========================================

    # 轉型態對齊契約
    df_candidates['stock_id'] = df_candidates['stock_id'].astype(str)
    df_candidates['name'] = df_candidates['name'].astype(str)
    df_candidates['industry'] = df_candidates['industry'].astype(str)
    df_candidates['market_cap'] = df_candidates['market_cap'].astype('float64')
    df_candidates['theme_tag'] = df_candidates['theme_tag'].astype(str)
    
    output_df = df_candidates[['stock_id', 'name', 'market_cap', 'industry', 'theme_tag']].dropna()
    output_df.to_parquet(os.path.join(processed_path, "candidates.parquet"), index=False)
    
    print(f"🎉 candidates.parquet 導出成功！共有 {len(output_df)} 檔純淨的個股候選標的。")

if __name__ == "__main__":
    generate_candidate_pool()