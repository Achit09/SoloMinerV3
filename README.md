# 加密貨幣 Solo 挖礦工具集

一個支持多種加密貨幣的 Solo 挖礦工具集合，包含 Dogecoin、Bitcoin Cash 和 Bitcoin-S 的挖礦程序。

## 支持的幣種

### 1. Dogecoin (DOGE)
- 檔案：`DogecoinMiner.py`
- 算法：Scrypt
- 特點：
  - 多核心 CPU 挖礦
  - 自動難度調整
  - 實時狀態顯示
  - 自動重連機制

### 2. Bitcoin Cash (BCH)
- 檔案：`SoloMinerBCH.py`
- 算法：SHA256D
- 特點：
  - 支持 BCH 協議
  - 區塊高度追蹤
  - 智能目標難度計算

### 3. Bitcoin-S
- 檔案：`SoloMinerBTCS.py`
- 算法：SHA256D
- 特點：
  - 支持 BTC-S 協議
  - 高效能 Share 提交
  - 優化的記憶體使用

### 4. 哈希優化模組
- 檔案：`hash_optimized.pyx`
- 功能：
  - 提供優化的哈希計算
  - Cython 加速支持
  - 用於提升挖礦效率

## 系統需求

- Python 3.8+
- 多核心 CPU
- 網絡連接
- 依賴套件：
  - pyscrypt
  - hashlib
  - multiprocessing

## 安裝

1. 克隆倉庫：
```bash
git clone https://github.com/yourusername/CryptoMiner.git
cd CryptoMiner
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### Dogecoin 挖礦
```bash
python DogecoinMiner.py
```
- 默認礦池：doge.luckymonster.pro:5112
- 挖礦模式：Solo
- 區塊獎勵：10000 DOGE

### Bitcoin Cash 挖礦
```bash
python SoloMinerBCH.py
```
- 支持 BCH 主網
- 自動難度調整

### Bitcoin-S 挖礦
```bash
python SoloMinerBTCS.py
```
- 支持 BTC-S 網絡
- 優化的挖礦算法

## 狀態顯示說明

- B: 當前區塊高度
- H: 哈希率
- S: 已找到的 Shares (每分鐘)
- D: 當前難度

## 配置說明

每個礦工程序都可以通過修改源碼中的配置部分來自定義：
- 礦池地址和端口
- 錢包地址
- CPU 核心使用數
- 批次大小
- 目標難度

## 性能優化

- 使用多進程進行哈希計算
- 優化的記憶體管理
- 智能的工作分配
- 自動的難度調整

## 注意事項

1. CPU 挖礦效率較低，建議僅用於學習和測試
2. 確保網絡連接穩定
3. 監控系統資源使用
4. 定期檢查挖礦狀態

## 貢獻

歡迎提交 Pull Requests 和 Issues。

## 許可證

MIT License

## 免責聲明

本程序僅供學習和研究使用，作者不對使用過程中造成的任何損失負責。
