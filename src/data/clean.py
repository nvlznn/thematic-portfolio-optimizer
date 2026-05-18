import os
import pandas as pd

def clean_and_melt_data():
    raw_path = "data/raw/202604"
    processed_path = "data/processed"
    
    # 確保輸出目錄存在
    os.makedirs(processed_path, exist_ok=True)
    
    # 讀取檔案的防爆函式
    def load_csv(filename):
        full_path = os.path.join(raw_path, filename)
        try:
            return pd.read_csv(full_path, encoding="cp950")
        except UnicodeDecodeError:
            return pd.read_csv(full_path, encoding="utf-8-sig")

    # 1. 讀取 4 張實際的 CSV 檔案
    df_prices = load_csv("raw_prices.csv")
    df_cap = load_csv("raw_market_cap.csv")
    df_beta = load_csv("raw_beta.csv")
    df_fin = load_csv("raw_financials.csv")
    
    
    # 2. 統一代碼與日期欄位名稱
    # 處理日頻率表
    for df in [df_prices, df_cap, df_beta]:
        df.rename(columns={'證券代碼': 'stock_id', '年月日': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str).str.strip()
        
    # 處理季頻率財務表 (從圖片中可見時間欄位為「年月」)
    df_fin.rename(columns={'證券代碼': 'stock_id', '年月': 'date'}, inplace=True)
    df_fin['date'] = pd.to_datetime(df_fin['date'])
    df_fin['stock_id'] = df_fin['stock_id'].astype(str).str.strip()

    # ==========================================
    # 建立 §7.1 prices.parquet (回測與流動性用)
    # ==========================================
    # 根據實際圖片欄位更名
    df_prices_clean = df_prices.rename(columns={
        '收盤價(元)': 'close',
        '成交量(千股)': 'volume',
        '成交值(千元)': 'amount'
    })
    
    # 單位轉換：千股 -> 股，千元 -> 元 (符合契約規定)
    df_prices_clean['volume'] = df_prices_clean['volume'] * 1000
    df_prices_clean['amount'] = df_prices_clean['amount'] * 1000
    
    # 確保數值型態正確
    df_prices_clean['close'] = pd.to_numeric(df_prices_clean['close'], errors='coerce')
    df_prices_clean['volume'] = pd.to_numeric(df_prices_clean['volume'], errors='coerce')
    df_prices_clean['amount'] = pd.to_numeric(df_prices_clean['amount'], errors='coerce')
    
    # 移移除大盤加權指數 Y9999，只保留個股放進主回測價格表
    df_stock_prices = df_prices_clean[df_prices_clean['stock_id'] != 'Y9999'].copy()
    
    # 嚴格符合 §7.1 介面契約欄位：date, stock_id, close, volume, amount
    prices_output = df_stock_prices[['date', 'stock_id', 'close', 'volume', 'amount']].dropna(subset=['close'])
    prices_output.to_parquet(os.path.join(processed_path, "prices.parquet"), index=False)

    # ==========================================
    # 建立 §7.2 candidates.parquet (候選股池與限制條件用)
    # ==========================================
    # 轉換市值欄位：市值(百萬元) -> 元
    df_cap.rename(columns={'市值(百萬元)': 'market_cap_raw'}, inplace=True)
    df_cap['market_cap'] = pd.to_numeric(df_cap['market_cap_raw'], errors='coerce') * 1000000
    
    # 取四月份最後一個交易日（最新一天）的市值作為模型輸入的個股市值
    latest_date = df_cap['date'].max()
    df_latest_cap = df_cap[df_cap['date'] == latest_date][['stock_id', 'market_cap']]
    
    # 從價格表中提取獨立的股票代碼與其對應的最新產業名稱 (TSE產業_名稱)
    df_industry = df_prices[df_prices['stock_id'] != 'Y9999'].sort_values('date').groupby('stock_id').last().reset_index()
    df_industry = df_industry[['stock_id', 'TSE產業_名稱']].rename(columns={'TSE產業_名稱': 'industry'})
    
    # 合併市值與產業分類
    df_cand_merged = pd.merge(df_industry, df_latest_cap, on='stock_id', how='left')
    
    # 根據契約補上 name (股票名稱) 與 theme_tag (主題標籤) 預設值
    df_cand_merged['name'] = "測試股票"
    df_cand_merged['theme_tag'] = "high_div"
    
    # 欄位排序嚴格符合 §7.2 介面契約：stock_id, name, market_cap, industry, theme_tag
    candidates_output = df_cand_merged[['stock_id', 'name', 'market_cap', 'industry', 'theme_tag']].dropna(subset=['market_cap'])
    candidates_output.to_parquet(os.path.join(processed_path, "candidates.parquet"), index=False)
    print("💾 §7.2 candidates.parquet 導出成功！")
    
    # ==========================================
    # 建立 params.parquet 供組員 B、C 算連續限制條件
    # ==========================================
    # 將連字號做清洗以防萬一
    df_fin.columns = [c.replace('－', '-').replace('—', '-') for c in df_fin.columns]
    
    df_beta_clean = df_beta.rename(columns={'CAPM_Beta 三月': 'beta_3m', 'CAPM_Beta 六月': 'beta_6m'})
    df_fin_clean = df_fin.rename(columns={'ROE(A)-稅後': 'roe', '營收成長率': 'revenue_growth'})
    
    # 合併日頻率參數 (股價 + 最新市值 + Beta)
    df_daily_params = pd.merge(prices_output, df_latest_cap, on='stock_id', how='left')
    df_daily_params = pd.merge(df_daily_params, df_beta_clean[['date', 'stock_id', 'beta_3m']], on=['date', 'stock_id'], how='left')
    
    # 合併季頻率財務資料並向下填充 (ffill)
    df_final_params = pd.merge(df_daily_params, df_fin_clean[['date', 'stock_id', 'roe', 'revenue_growth']], on=['date', 'stock_id'], how='left')
    df_final_params.sort_values(by=['stock_id', 'date'], inplace=True)
    df_final_params[['roe', 'revenue_growth']] = df_final_params.groupby('stock_id')[['roe', 'revenue_growth']].ffill()
    
    df_final_params.to_parquet(os.path.join(processed_path, "params.parquet"), index=False)
    
    print("\n prices.parquet 前五行預覽：")
    print(prices_output.head())
    print("\n candidates.parquet 前五行預覽：")
    print(candidates_output.head())

if __name__ == "__main__":
    clean_and_melt_data()