# 挖礦上下文配置

# 訂閱詳情
sub_details = None
extranonce1 = None
extranonce2_size = None
extranonce2 = None

# 工作參數
job_id = None
prevhash = None
coinb1 = None
coinb2 = None
merkle_branch = []
version = None
nbits = None
ntime = None
clean_jobs = True

# 狀態標記
fShutdown = False
updatedPrevHash = None

# 線程運行狀態
listfThreadRunning = [False] * 2  # [miner_thread, subscribe_thread]

# 區塊難度記錄
nHeightDiff = {}  # 記錄每個高度的最佳難度

# Mining stats
difficulty = 0
target = None

def reset():
    """重置所有變量"""
    global sub_details, extranonce1, extranonce2_size, extranonce2
    global job_id, prevhash, coinb1, coinb2, merkle_branch
    global version, nbits, ntime, clean_jobs
    global fShutdown, updatedPrevHash
    
    sub_details = None
    extranonce1 = None
    extranonce2_size = None
    extranonce2 = None
    
    job_id = None
    prevhash = None
    coinb1 = None
    coinb2 = None
    merkle_branch = []
    version = None
    nbits = None
    ntime = None
    clean_jobs = True
    
    fShutdown = False
    updatedPrevHash = None
    
    listfThreadRunning[0] = False
    listfThreadRunning[1] = False
    
    nHeightDiff.clear() 