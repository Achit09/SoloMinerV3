import binascii
import hashlib
import json
import logging
import random
import socket
import threading
import time
import traceback
from datetime import datetime
from signal import SIGINT , signal
import os
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
import shutil

import requests
from colorama import init, Fore, Back, Style

import context as ctx

try:
    import pyopencl as cl
    
    def opencl_hash_calculation(blockheader, nonce_start, batch_size=1000):
        """使用 OpenCL 加速哈希計算"""
        # 初始化 OpenCL
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        # OpenCL 內核代碼
        kernel_code = """
        __kernel void sha256_hash(__global const uchar* header,
                                __global uint* nonces,
                                __global ulong* results,
                                const ulong target) {
            int gid = get_global_id(0);
            // SHA256 計算邏輯
            // ...
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        
        # 準備數據
        header_bin = np.frombuffer(binascii.unhexlify(blockheader[:-8]), dtype=np.uint8)
        nonces = np.arange(nonce_start, nonce_start + batch_size, dtype=np.uint32)
        
        # 分配緩衝區
        header_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header_bin)
        nonces_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=nonces)
        results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, nonces.nbytes)
        
        # 執行內核
        program.sha256_hash(queue, (batch_size,), None, header_buf, nonces_buf, results_buf)
        
        # 獲取結果
        results = np.empty_like(nonces)
        cl.enqueue_copy(queue, results, results_buf)
        
        return process_results(results, target)
except ImportError:
    pass

try:
    from numba import jit, uint32, uint64
    import numpy as np
    
    @jit(nopython=True, parallel=True)
    def numba_hash_calculation(header_bytes, nonces, target):
        """使用 Numba 加速的哈希計算"""
        results = []
        for nonce in nonces:
            full_header = header_bytes + nonce.tobytes()
            hash1 = hashlib.sha256(full_header).digest()
            hash2 = hashlib.sha256(hash1).digest()
            hash_int = int.from_bytes(hash2, byteorder='little')
            
            if hash_int <= target:
                results.append((nonce, hash2.hex()))
                if len(results) >= 2:
                    break
        return results
except ImportError:
    pass

sock = None

# 在全局變量區域添加
total_shares = 0
rejected_shares = 0  # 添加拒絕計數
blocks_found = 0
current_worker_name = None
mining_socket = None  # 改名以避免與參數衝突

# 在全局變量區域添加礦池配置
POOL_CONFIG = {
    'url': 'btc.viabtc.io',
    'ports': [3333, 443, 25],  # 調整端口順序
    'worker_name': 'Achit999',
    'password': 'x',
    'protocol': 'stratum+tcp',
    'backup_urls': [  # 添加備用礦池地址
        'btc.viabtc.com',
        'btc-pool.viabtc.com'
    ]
}

# 在程式開始時初始化 colorama
init(autoreset=True, convert=True)

# 添加全局變量來存儲最後兩條消息和狀態
LAST_MESSAGES = []
MAX_MESSAGES = 2
DISPLAY_LOCK = threading.Lock()  # 添加鎖來同步顯示

# 添加全局變量來存儲錯誤消息
ERROR_MESSAGES = []
MAX_ERROR_MESSAGES = 5  # 保留最近5條錯誤消息

# 添加全局變量來控制顯示更新
LAST_STATUS = ""
TERMINAL_HEIGHT = 24  # 默認終端高度

ML_AVAILABLE = False
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    logg("[ML] scikit-learn not available, running without ML optimization")

def timer() :
    tcx = datetime.now().time()
    return tcx

def load_wallet_address():
    """從配置文件讀取錢包地址"""
    config_path = 'config.txt'
    default_address = 'bc1qmvplzwalslgmeavt525ah6waygkrk99gpc22hj'
    
    try:
        if not os.path.exists(config_path):
            # 如果配置文件不存在，創建一個
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("# Bitcoin Wallet Address Configuration\n")
                f.write("# Please enter your BTC wallet address below:\n")
                f.write(f"BTC_WALLET={default_address}\n")
            logg("[*] Created new config file with default wallet address")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if 'BTC_WALLET=' in line:
                        address = line.split('=')[1].strip()
                        if address:
                            return address
                            
        logg("[!] No valid wallet address found in config file, using default")
        return default_address
        
    except Exception as e:
        logg(f"[!] Error reading config file: {str(e)}")
        return default_address

# 替換原有的地址定義
address = load_wallet_address()

print(Back.BLUE , Fore.WHITE , 'BTC WALLET:' , Fore.BLACK , str(address) , Style.RESET_ALL)


def handler(signal_received , frame) :
    # Handle any cleanup here
    ctx.fShutdown = True
    print(Fore.MAGENTA , '[' , timer() , ']' , Fore.YELLOW , 'Terminating Miner, Please Wait..')


def get_resource_path(relative_path):
    """獲取資源文件的絕對路徑"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 創建臨時文件夾，將路徑存儲在 _MEIPASS 中
        base_path = sys._MEIPASS
    elif 'Contents/Resources' in os.path.abspath(__file__):
        # py2app 打包後的路徑
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def get_log_path():
    """獲取日誌文件路徑"""
    if os.name == 'nt':  # Windows
        log_path = os.path.join(os.getenv('LOCALAPPDATA'), 'ViaBTCMiner', 'Logs')
    else:  # macOS/Linux
        log_path = os.path.expanduser('~/Library/Logs/ViaBTCMiner')
        
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # 主日誌文件路徑
    current_log = os.path.join(log_path, 'miner.log')
    backup_log = os.path.join(log_path, 'miner.log.1')
    
    return current_log, backup_log

def rotate_logs(current_log, backup_log):
    """輪換日誌文件"""
    try:
        if os.path.exists(current_log):
            # 如果存在備份文件，刪除它
            if os.path.exists(backup_log):
                os.remove(backup_log)
            # 將當前日誌改名為備份
            os.rename(current_log, backup_log)
    except Exception as e:
        print(f"Error rotating logs: {str(e)}")

def print_messages():
    """打印狀態欄和錯誤消息，保留所有信息"""
    global LAST_STATUS
    with DISPLAY_LOCK:
        # 構建狀態欄
        if hasattr(print_messages, 'last_status'):
            status_bar = [
                "=" * shutil.get_terminal_size().columns,
                print_messages.last_status,
                "=" * shutil.get_terminal_size().columns
            ]
            # 直接打印狀態欄，不清屏
            print("\n".join(status_bar))
        
        # 打印錯誤消息（如果有新的）
        if ERROR_MESSAGES:
            new_status = "\n".join([
                "Error messages:",
                "!" * 80,
                *[f"{Fore.RED}{msg}{Style.RESET_ALL}" for msg in ERROR_MESSAGES[-5:]],
                "!" * 80
            ])
            if new_status != LAST_STATUS:
                print("\n" + new_status)
                LAST_STATUS = new_status

def logg(msg):
    """改進的日誌記錄函數，直接打印並記錄到文件"""
    current_log, backup_log = get_log_path()
    
    # 檢查當前日誌文件大小
    if os.path.exists(current_log) and os.path.getsize(current_log) > 1024 * 1024:  # 1MB
        rotate_logs(current_log, backup_log)
    
    # 配置日誌
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(current_log),
            logging.StreamHandler()
        ]
    )
    
    # 記錄日誌並直接打印
    message = str(msg)
    logging.info(message)
    print(f"{Fore.WHITE}[INFO] {message}{Style.RESET_ALL}")
    
    # 如果是 JSON 相關錯誤，記錄更多詳細信息
    if 'JSON' in message or 'json' in message:
        logging.debug(f"JSON Error Details - Raw message: {msg}")
        if isinstance(msg, Exception):
            logging.debug(f"Exception type: {type(msg)}")
            logging.debug(f"Traceback: {traceback.format_exc()}")
    
    # 更新狀態欄
    print_messages()

def log_error(error_msg):
    """記錄錯誤消息並立即更新顯示"""
    global ERROR_MESSAGES
    error_msg = f"[ERROR] {error_msg}"
    ERROR_MESSAGES.append(error_msg)
    if len(ERROR_MESSAGES) > MAX_ERROR_MESSAGES:
        ERROR_MESSAGES.pop(0)
    logg(error_msg)
    # 立即更新顯示
    print_messages()

def get_current_block_height() :
    # returns the current network height
    r = requests.get('https://blockchain.info/latestblock')
    return int(r.json()['height'])


def check_for_shutdown(t) :
    # handle shutdown
    n = t.n
    if ctx.fShutdown :
        if n != -1 :
            ctx.listfThreadRunning[n] = False
            t.exit = True


class ExitedThread(threading.Thread) :
    def __init__(self , arg , n) :
        super(ExitedThread , self).__init__()
        self.exit = False
        self.arg = arg
        self.n = n

    def run(self) :
        self.thread_handler(self.arg , self.n)
        pass

    def thread_handler(self , arg , n) :
        while True :
            check_for_shutdown(self)
            if self.exit :
                break
            ctx.listfThreadRunning[n] = True
            try :
                self.thread_handler2(arg)
            except Exception as e :
                logg("ThreadHandler()")
                print(Fore.MAGENTA , '[' , timer() , ']' , Fore.WHITE , 'ThreadHandler()')
                logg(e)
                print(Fore.RED , e)
            ctx.listfThreadRunning[n] = False

            time.sleep(2)
            pass

    def thread_handler2(self , arg) :
        raise NotImplementedError("must impl this func")

    def check_self_shutdown(self) :
        check_for_shutdown(self)

    def try_exit(self) :
        self.exit = True
        ctx.listfThreadRunning[self.n] = False
        pass


def optimize_hash_calculation(blockheader, nonce_start, batch_size=1000):
    """優化的 PPS+ 哈希計算，降低難度要求"""
    try:
        header_bin = binascii.unhexlify(blockheader[:-8])
        nonces = np.arange(nonce_start, nonce_start + batch_size, dtype=np.uint32)
        results = []
        
        # 使用更寬鬆的難度計算
        if hasattr(ctx, 'target'):
            # 使用 ViaBTC 的難度，但降低要求
            pool_diff = float(ctx.target)
            adjusted_diff = pool_diff / 4  # 降低難度要求
            target = int(0x00000000FFFF0000000000000000000000000000000000000000000000000000 / adjusted_diff)
        else:
            # 使用非常寬鬆的默認目標值
            target = int(0x00000000FFFF0000000000000000000000000000000000000000000000000000 / 4096)
        
        # 使用向量化運算
        headers = np.array([header_bin + nonce.tobytes() for nonce in nonces])
        hash1 = np.array([hashlib.sha256(h).digest() for h in headers])
        hash2 = np.array([hashlib.sha256(h).digest() for h in hash1])
        hash_ints = np.array([int.from_bytes(h, byteorder='little') for h in hash2])
        
        # 找到所有符合條件的 nonce
        valid_indices = np.where(hash_ints <= target)[0]
        
        if len(valid_indices) > 0:
            # 按哈希值排序，選擇最好的幾個
            sorted_indices = valid_indices[np.argsort(hash_ints[valid_indices])]
            for idx in sorted_indices[:2]:  # 每次最多提交2個最好的
                results.append((int(nonces[idx]), binascii.hexlify(hash2[idx]).decode()))
        
        return results, nonce_start + batch_size
        
    except Exception as e:
        log_error(f"Hash calculation error: {str(e)}")
        return [], nonce_start + batch_size

def generate_worker_name():
    """生成礦工名稱：Achit999.XX"""
    worker_id = random.randint(1, 99)
    return f"{POOL_CONFIG['worker_name']}.{worker_id:02d}"  # 使用 Achit999.XX 格式

def submit_shares(shares, ctx, connection, worker_name):
    """優化的 Share 提交策略，提高接受率"""
    global total_shares, rejected_shares
    successful_shares = 0
    
    try:
        for nonce, hash_hex in shares:
            try:
                # 構建提交數據
                payload = {
                    "id": 4,
                    "method": "mining.submit",
                    "params": [
                        worker_name,
                        ctx.job_id,
                        ctx.extranonce2,
                        ctx.ntime,
                        hex(nonce)[2:].zfill(8)
                    ]
                }
                
                # 提交前等待較短時間
                time.sleep(0.05)  # 減少等待時間
                connection.sendall(json.dumps(payload).encode() + b'\n')
                
                # 等待響應
                response = connection.recv(1024).decode().strip()
                response_data = json.loads(response)
                
                if response_data.get('result') is True:
                    successful_shares += 1
                    total_shares += 1
                    logg(f"[*] Share accepted! Total accepted: {total_shares}")
                else:
                    error = response_data.get('error')
                    if error:
                        rejected_shares += 1
                        error_code = error[0] if isinstance(error, list) else None
                        
                        if error_code == 20:  # 無效的 share
                            logg(f"[!] Invalid share: {error[1] if len(error) > 1 else 'unknown error'}")
                            time.sleep(0.5)
                        elif error_code == 21:  # 重複的 share
                            logg("[!] Duplicate share")
                            time.sleep(0.2)
                        elif error_code == 10:  # 頻率限制
                            logg("[!] Rate limit exceeded")
                            time.sleep(1)
                        else:
                            logg(f"[!] Share rejected: {error}")
                            time.sleep(0.5)
                
            except Exception as e:
                log_error(f"Share submission error: {str(e)}")
                time.sleep(0.5)
                continue
        
        return successful_shares
        
    except Exception as e:
        log_error(f"Share submission failed: {str(e)}")
        return 0

def adjust_difficulty(hashrate):
    """更激進的難度調整"""
    if hashrate > 1e6:  # > 1 MH/s
        return 64
    elif hashrate > 1e5:  # > 100 KH/s
        return 32
    elif hashrate > 1e4:  # > 10 KH/s
        return 16
    else:
        return 8  # 使用最低難度

def adjust_target_difficulty(current_hashrate):
    """根據算力動態調整目標值"""
    if current_hashrate > 50000:  # > 50 KH/s
        return 1.2
    elif current_hashrate > 20000:  # > 20 KH/s
        return 1.5
    elif current_hashrate > 10000:  # > 10 KH/s
        return 2
    else:
        return 3

def update_mining_difficulty(sock, worker_name, current_hashrate):
    """使用更保守的難度設置"""
    try:
        # 根據算力設置合適的難度
        if current_hashrate > 1e6:  # > 1 MH/s
            new_diff = 256
        elif current_hashrate > 1e5:  # > 100 KH/s
            new_diff = 128
        elif current_hashrate > 2e4:  # > 20 KH/s
            new_diff = 64
        else:
            new_diff = 32
        
        # 發送難度更新請求
        diff_payload = {
            "id": 3,
            "method": "mining.suggest_difficulty",
            "params": [new_diff]
        }
        sock.sendall(json.dumps(diff_payload).encode() + b'\n')
        
        logg(f"[*] Suggested new difficulty: {new_diff}")
        
    except Exception as e:
        logg(f"[!] Failed to update difficulty: {str(e)}")

class MiningOptimizer:
    _instance = None
    
    def __new__(cls):
        """實現單例模式"""
        if cls._instance is None:
            cls._instance = super(MiningOptimizer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化挖礦優化器"""
        if self.initialized:
            return
            
        self.history = []
        self.trained = False
        self.min_samples = 500
        self.model = None
        self.scaler = None
        self.ml_enabled = ML_AVAILABLE
        
        if self.ml_enabled:
            try:
                self.scaler = StandardScaler()
                logg("[ML] Mining optimizer initialized")
            except Exception as e:
                log_error(f"Failed to initialize ML model: {str(e)}")
                self.ml_enabled = False
                
        self.initialized = True
    
    def _init_model(self):
        """延遲初始化模型"""
        if not self.model and self.ml_enabled:
            try:
                self.model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    n_jobs=-1
                )
                return True
            except Exception as e:
                log_error(f"Model initialization error: {str(e)}")
                return False
        return bool(self.model)
    
    def predict_success_probability(self, nonce, hash_int):
        """預測成功概率"""
        if not self.ml_enabled or not self.trained:
            return 0.5
            
        try:
            if not self._init_model():
                return 0.5
                
            if len(self.history) < self.min_samples:
                return 0.5
                
            X = np.array([[nonce, hash_int]])
            X = self.scaler.transform(X)
            return float(self.model.predict(X)[0])
        except Exception as e:
            log_error(f"Prediction error: {str(e)}")
            return 0.5
    
    def train_model(self):
        """訓練模型"""
        if not self.ml_enabled or len(self.history) < self.min_samples:
            return False
            
        try:
            if not self._init_model():
                return False
                
            X = np.array([[d['nonce'], d['hash_int']] for d in self.history])
            y = np.array([d['success'] for d in self.history])
            
            X = self.scaler.fit_transform(X)
            self.model.fit(X, y)
            self.trained = True
            logg("[ML] Model training completed")
            return True
            
        except Exception as e:
            log_error(f"Model training error: {str(e)}")
            self.trained = False
            return False
    
    def collect_data(self, nonce, hash_int, success):
        """收集挖礦數據"""
        try:
            if len(self.history) >= 10000:
                self.history = self.history[1000:]
                
            self.history.append({
                'nonce': nonce,
                'hash_int': hash_int,
                'success': 1 if success else 0
            })
            
            if len(self.history) >= self.min_samples and not self.trained:
                self.train_model()
                
        except Exception as e:
            log_error(f"Data collection error: {str(e)}")

def simulate_asic_behavior(blockheader, nonce_start, batch_size=1000):
    """結合機器學習的挖礦模擬"""
    try:
        results = []
        base_nonce = nonce_start & 0xFFFFFFFF
        optimizer = MiningOptimizer()
        
        for i in range(batch_size):
            current_nonce = (base_nonce + i) & 0xFFFFFFFF
            
            # 獲取哈希結果
            hash_result, hash_int = generate_base_hash(
                blockheader, 
                current_nonce,
                ctx.job_id
            )
            
            # 收集數據
            success = 1 if hash_result else 0
            optimizer.collect_data(current_nonce, hash_int, success)
            
            # 定期訓練模型
            if i % 1000 == 0:
                optimizer.train_model()
            
            if hash_result:
                # 使用模型預測成功概率
                prob = optimizer.predict_success_probability(current_nonce, hash_int)
                if prob > 0.6:  # 只保留高概率的結果
                    results.append((current_nonce, hash_result))
                    logg(f"[ML] High probability share found: {prob:.2f}")
                
                if len(results) >= 5:
                    break
        
        # 返回結果和下一個nonce
        return results, (base_nonce + batch_size) & 0xFFFFFFFF
        
    except Exception as e:
        log_error(f"Mining simulation error: {str(e)}")
        return [], (base_nonce + batch_size) & 0xFFFFFFFF

def generate_base_hash(blockheader, nonce, job_id):
    """優化的哈希生成"""
    try:
        nonce = nonce & 0xFFFFFFFF
        header_hex = blockheader[:-8]
        header_bin = binascii.unhexlify(header_hex)
        nonce_bytes = nonce.to_bytes(4, byteorder='little')
        full_header = header_bin + nonce_bytes
        
        # 計算哈希
        hash1 = hashlib.sha256(full_header).digest()
        hash2 = hashlib.sha256(hash1).digest()
        hash_int = int.from_bytes(hash2, 'little')
        
        if hasattr(ctx, 'target'):
            target_int = int(ctx.target)
            # 使用動態難度調整
            difficulty_multiplier = 16
            if MiningOptimizer().trained:
                difficulty_multiplier = 8  # 使用較嚴格的標準
                
            if hash_int <= target_int * difficulty_multiplier:
                return hash2.hex(), hash_int
        return None, hash_int
        
    except Exception as e:
        log_error(f"Hash generation error: {str(e)}")
        return None, None

def monitor_pool_messages(connection):
    """監聽礦池消息"""
    try:
        data = connection.recv(4096).decode().strip()
        if data:
            messages = data.split('\n')
            for message in messages:
                try:
                    msg = json.loads(message)
                    if 'method' in msg:
                        if msg['method'] == 'mining.notify':
                            logg(f"[Pool] New work received - Job ID: {msg['params'][0]}")
                        elif msg['method'] == 'mining.set_difficulty':
                            logg(f"[Pool] Difficulty changed to: {msg['params'][0]}")
                    elif 'result' in msg:
                        if msg.get('error'):
                            logg(f"[Pool] Error: {msg['error']}")
                        else:
                            logg(f"[Pool] Response: {msg}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log_error(f"Pool monitoring error: {str(e)}")

def verify_share_locally(nonce, hash_hex, target):
    """本地驗證 share"""
    try:
        hash_int = int(hash_hex, 16)
        target_int = int(target, 16)
        
        if hash_int <= target_int:
            logg(f"[Verify] Valid share found - Hash: {hash_hex[:20]}...")
            logg(f"[Verify] Target: {target[:20]}...")
            logg(f"[Verify] Nonce: {hex(nonce)}")
            return True
        else:
            logg(f"[Verify] Invalid share - Hash too high")
            return False
    except Exception as e:
        log_error(f"Share verification error: {str(e)}")
        return False

def submit_asic_shares(shares, ctx, connection, worker_name):
    """改進的 share 提交和監控"""
    global total_shares, rejected_shares
    successful_shares = 0
    
    try:
        last_submit_time = time.time()
        
        for nonce, hash_hex in shares:
            try:
                # 本地驗證
                if not verify_share_locally(nonce, hash_hex, ctx.target):
                    logg("[Submit] Share failed local verification, skipping...")
                    continue
                
                # 提交間隔控制
                current_time = time.time()
                elapsed = current_time - last_submit_time
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)
                
                # 詳細的提交信息
                logg(f"[Submit] Attempting share submission:")
                logg(f"[Submit] Nonce: {hex(nonce)}")
                logg(f"[Submit] Job ID: {ctx.job_id}")
                logg(f"[Submit] ExtraNonce2: {ctx.extranonce2}")
                logg(f"[Submit] NTime: {ctx.ntime}")
                
                payload = {
                    "id": 4,
                    "method": "mining.submit",
                    "params": [
                        worker_name,
                        ctx.job_id,
                        ctx.extranonce2,
                        ctx.ntime,
                        hex(nonce)[2:].zfill(8)
                    ]
                }
                
                # 發送和監控
                connection.sendall(json.dumps(payload).encode() + b'\n')
                monitor_pool_messages(connection)
                
                response = connection.recv(4096).decode().strip()
                if response:
                    response_data = json.loads(response)
                    if response_data.get('result') is True:
                        successful_shares += 1
                        total_shares += 1
                        logg(f"[Success] Share accepted ({total_shares})")
                        logg(f"[Success] Hash: {hash_hex[:20]}...")
                    else:
                        rejected_shares += 1
                        error = response_data.get('error')
                        if error:
                            logg(f"[Reject] Share rejected - Error: {str(error)}")
                            logg(f"[Reject] Hash: {hash_hex[:20]}...")
                
                last_submit_time = time.time()
                
            except Exception as e:
                log_error(f"Share submission error: {str(e)}")
                time.sleep(0.1)
                continue
        
        return successful_shares
        
    except Exception as e:
        log_error(f"Share submission failed: {str(e)}")
        return 0

def bitcoin_miner(t, restarted=False):
    """比特幣挖礦主函數"""
    global total_shares, rejected_shares, work_on
    
    try:
        # 初始化變量
        nonce_start = random.randint(0, 2**32 - 1)
        total_hashes = 0
        last_report_time = time.time()
        report_interval = 1.0
        batch_size = 1000  # 定義批次大小
        
        # 驗證必要參數
        if not all([ctx.nbits, ctx.extranonce2_size, ctx.coinb1, ctx.extranonce1, ctx.coinb2]):
            logg("[!] Missing required parameters, waiting...")
            time.sleep(5)
            return False
            
        # 生成 extranonce2
        ctx.extranonce2 = hex(random.randint(0, 2**32 - 1))[2:].zfill(2 * ctx.extranonce2_size)
        
        # 初始構建 coinbase
        coinbase = create_coinbase()
        work_on = get_current_block_height()
        
        logg(f'[*] Working on block height: {work_on + 1}')
        
        while True:
            try:
                # 檢查是否需要更新工作
                if ctx.updatedPrevHash:
                    coinbase = create_coinbase()
                    ctx.updatedPrevHash = False
                    work_on = get_current_block_height()
                    logg(f'[*] New work received, height: {work_on + 1}')
                
                # 記錄開始時間
                start_time = time.time()
                
                # 模擬ASIC挖礦
                results, nonce_start = simulate_asic_behavior(
                    coinbase,
                    nonce_start,
                    batch_size=batch_size
                )
                
                # 計算實際哈希數
                total_hashes += batch_size
                
                # 提交shares
                if results:
                    successful = submit_asic_shares(results, ctx, mining_socket, current_worker_name)
                    if successful:
                        logg(f"[Mining] Successfully submitted {successful} shares")
                
                # 更新顯示
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed = current_time - last_report_time
                    hashrate = total_hashes / elapsed if elapsed > 0 else 0
                    print_status_bar(hashrate, total_shares, work_on + 1)
                    last_report_time = current_time
                    total_hashes = 0  # 重置哈希計數
                
                # 控制挖礦速度
                elapsed = time.time() - start_time
                if elapsed < 0.001:  # 確保每批次至少花費1ms
                    time.sleep(0.001 - elapsed)
                
            except socket.error as e:
                log_error(f"Socket error: {str(e)}")
                if not restarted:
                    return bitcoin_miner(t, True)
                time.sleep(1)
                continue
                
            except Exception as e:
                log_error(f"Mining error: {str(e)}")
                time.sleep(1)
                continue
                
    except Exception as e:
        log_error(f"Fatal mining error: {str(e)}")
        return False

def create_coinbase():
    """創建 coinbase 交易"""
    try:
        if not all([ctx.coinb1, ctx.extranonce1, ctx.extranonce2, ctx.coinb2]):
            raise ValueError("Missing coinbase parameters")
            
        return ctx.coinb1 + ctx.extranonce1 + ctx.extranonce2 + ctx.coinb2
        
    except Exception as e:
        log_error(f"Coinbase creation error: {str(e)}")
        raise

def block_listener(t):
    """處理礦池連接和消息監聽"""
    global mining_socket, current_worker_name
    max_retries = 5
    retry_count = 0
    current_port_index = 0
    
    while retry_count < max_retries:
        try:
            mining_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            mining_socket.settimeout(30)
            
            port = POOL_CONFIG['ports'][current_port_index]
            logg(f"[*] Connecting to pool {POOL_CONFIG['url']}:{port}")
            
            try:
                mining_socket.connect((POOL_CONFIG['url'], port))
            except socket.error as e:
                raise ConnectionError(f"Connection failed: {str(e)}")
            
            # 發送訂閱請求
            subscribe_req = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            mining_socket.sendall(json.dumps(subscribe_req).encode() + b'\n')
            
            # 處理響應
            buffer = ""
            subscription_complete = False
            authorization_complete = False
            
            while not (subscription_complete and authorization_complete):
                data = mining_socket.recv(4096).decode()
                if not data:
                    raise ConnectionError("Connection closed")
                
                buffer += data
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    
                    try:
                        response = json.loads(line)
                        
                        # 處理訂閱響應
                        if not subscription_complete and response.get('id') == 1:
                            if 'result' in response:
                                result = response['result']
                                if isinstance(result, list) and len(result) >= 3:
                                    ctx.sub_details = result[0]
                                    ctx.extranonce1 = result[1]
                                    ctx.extranonce2_size = result[2]
                                    logg("[*] Successfully subscribed")
                                    subscription_complete = True
                                    
                                    # 發送授權請求
                                    auth_req = {
                                        "id": 2,
                                        "method": "mining.authorize",
                                        "params": [current_worker_name, POOL_CONFIG['password']]
                                    }
                                    mining_socket.sendall(json.dumps(auth_req).encode() + b'\n')
                            
                        # 處理授權響應
                        elif not authorization_complete and response.get('id') == 2:
                            if response.get('result') is True:
                                logg("[*] Authorization successful")
                                authorization_complete = True
                            else:
                                raise ConnectionError("Authorization failed")
                        
                        # 處理難度設置
                        elif response.get('method') == 'mining.set_difficulty':
                            difficulty = response['params'][0]
                            ctx.target = difficulty
                            logg(f"[*] Difficulty set to {difficulty}")
                        
                        # 處理工作通知
                        elif response.get('method') == 'mining.notify':
                            params = response['params']
                            if len(params) >= 8:
                                ctx.job_id = params[0]
                                ctx.prevhash = params[1]
                                ctx.coinb1 = params[2]
                                ctx.coinb2 = params[3]
                                ctx.merkle_branch = params[4]
                                ctx.version = params[5]
                                ctx.nbits = params[6]
                                ctx.ntime = params[7]
                                ctx.clean_jobs = params[8] if len(params) > 8 else True
                                logg("[*] New work received")
                                
                    except json.JSONDecodeError as e:
                        logg(f"[DEBUG] JSON parse error: {str(e)}, raw data: {line}")
                        continue
            
            # 如果成功完成訂閱和授權，開始監聽工作
            if subscription_complete and authorization_complete:
                logg("[*] Connection established, starting mining operations")
                return True
            
        except Exception as e:
            retry_count += 1
            current_port_index = (current_port_index + 1) % len(POOL_CONFIG['ports'])
            
            logg(f"[!] Connection attempt {retry_count} failed: {str(e)}")
            
            if mining_socket:
                try:
                    mining_socket.close()
                except:
                    pass
            
            if retry_count < max_retries:
                wait_time = min(retry_count * 5, 30)
                logg(f"[*] Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
    
    logg("[!] Max retries reached, giving up")
    return False

def handle_mining_messages(socket):
    """處理礦池消息"""
    global current_worker_name
    buffer = ""
    socket.settimeout(30)  # 增加超時時間
    
    try:
        # 首先處理已經收到的工作通知
        if hasattr(ctx, 'job_id') and ctx.job_id:
            logg("[*] Starting with existing work")
            # 啟動挖礦線程
            miner_t = CoinMinerThread(None)
            miner_t.start()
        
        # 繼續監聽新消息
        while True:
            try:
                data = socket.recv(4096).decode()
                if not data:
                    raise ConnectionError("Connection closed by pool")
                
                buffer += data
                
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    message = message.strip()
                    
                    if not message:
                        continue
                    
                    try:
                        response = json.loads(message)
                        
                        # 處理不同類型的消息
                        if response.get('method') == 'mining.notify':
                            # 處理工作通知
                            params = response['params']
                            ctx.job_id = params[0]
                            ctx.prevhash = params[1]
                            ctx.coinb1 = params[2]
                            ctx.coinb2 = params[3]
                            ctx.merkle_branch = params[4]
                            ctx.version = params[5]
                            ctx.nbits = params[6]
                            ctx.ntime = params[7]
                            ctx.clean_jobs = params[8] if len(params) > 8 else True
                            
                            logg("[*] Received new work")
                            
                        elif response.get('method') == 'mining.set_difficulty':
                            difficulty = response['params'][0]
                            logg(f"[*] Difficulty set to {difficulty}")
                            ctx.target = difficulty
                            
                        elif response.get('error'):
                            logg(f"[!] Error from pool: {response['error']}")
                            
                    except json.JSONDecodeError as je:
                        logg(f"[!] JSON parse error: {str(je)}")
                        logg(f"[DEBUG] Problematic message: {message}")
                        continue
                    
            except socket.timeout:
                logg("[*] No new messages for 30 seconds, checking connection...")
                # 發送空閒保持連接
                try:
                    socket.sendall(b'{"id": 3, "method": "mining.noop", "params": []}\n')
                except (socket.error, ConnectionError) as e:
                    raise ConnectionError("Connection lost during idle") from e
                
            except (socket.error, ConnectionError) as e:
                logg(f"[!] Connection error: {str(e)}")
                return False
                
    except (socket.error, ConnectionError, json.JSONDecodeError) as e:
        logg(f"[!] Fatal error in message handler: {str(e)}")
        return False
    except Exception as e:
        logg(f"[!] Unexpected error in message handler: {str(e)}")
        return False
    
    return True

class CoinMinerThread(ExitedThread) :
    def __init__(self , arg = None) :
        super(CoinMinerThread , self).__init__(arg , n = 0)

    def thread_handler2(self , arg) :
        self.thread_bitcoin_miner(arg)

    def thread_bitcoin_miner(self , arg) :
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try :
            ret = bitcoin_miner(self)
            logg(Fore.MAGENTA , "[" , timer() , "] [*] Miner returned %s\n\n" % "true" if ret else "false")
            print(Fore.LIGHTCYAN_EX , "[*] Miner returned %s\n\n" % "true" if ret else "false")
        except Exception as e :
            logg("[*] Miner()")
            print(Back.WHITE , Fore.MAGENTA , "[" , timer() , "]" , Fore.BLUE , "[*] Miner()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

    pass


class NewSubscribeThread(ExitedThread) :
    def __init__(self , arg = None) :
        super(NewSubscribeThread , self).__init__(arg , n = 1)

    def thread_handler2(self , arg) :
        self.thread_new_block(arg)

    def thread_new_block(self , arg) :
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try :
            ret = block_listener(self)
        except Exception as e :
            logg("[*] Subscribe thread()")
            print(Fore.MAGENTA , "[" , timer() , "]" , Fore.YELLOW , "[*] Subscribe thread()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

    pass


def StartMining() :
    global current_worker_name  # 使用全局變量
    
    # 程序啟動時生成礦工名稱
    current_worker_name = generate_worker_name()
    logg(f"[*] Started mining with worker name: {current_worker_name}")
    print(Fore.CYAN, f"[{timer()}] Started mining with worker name: {current_worker_name}")
    
    clear_terminal()  # 啟動時清空終端
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            subscribe_t = NewSubscribeThread(None)
            subscribe_t.start()
            logg("[*] Subscribe thread started.")
            print(Fore.MAGENTA, "[", timer(), "]", Fore.GREEN, "[*] Subscribe thread started.")
            
            # 等待確保訂閱成功，增加等待時間
            time.sleep(6)
            
            # 驗證連接狀態
            if not mining_socket or not hasattr(mining_socket, 'fileno') or mining_socket.fileno() == -1:
                raise Exception("Socket connection failed")
            
            # 驗證必要參數
            if not all([ctx.nbits, ctx.job_id, ctx.prevhash]):
                raise Exception("Missing required mining parameters")
            
            logg("[*] Connection established successfully")
            print(Fore.GREEN, f"[{timer()}] Connection established successfully")
            
            miner_t = CoinMinerThread(None)
            miner_t.start()
            logg("[*] Bitcoin Miner Thread Started")
            print(Fore.MAGENTA, "[", timer(), "]", Fore.GREEN, "[*] Bitcoin Miner Thread Started")
            print(Fore.BLUE, '--------------~~( ', Fore.YELLOW, 'PC Lucky Miner', Fore.BLUE, ' )~~--------------')
            break
            
        except Exception as e:
            retry_count += 1
            logg(f"[!] Mining start failed (attempt {retry_count}/{max_retries}): {str(e)}")
            print(Fore.RED, f"[{timer()}] Mining start failed (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count < max_retries:
                time.sleep(5)
            else:
                logg("[!] Failed to start mining after maximum retries")
                print(Fore.RED, f"[{timer()}] Failed to start mining after maximum retries")
                return


def monitor_performance():
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    
    logg(f"[*] CPU Usage: {cpu_percent}%")
    logg(f"[*] Memory Usage: {memory_info.percent}%")


def clear_terminal():
    """清空終端"""
    if os.name == 'nt':  # Windows
        # 使用 ANSI 轉義序列而不是 cls
        print('\033[2J\033[H', end='')
    else:  # macOS/Linux
        os.system('clear')

def print_status_bar(hashrate, shares, height):
    """改進的狀態顯示"""
    try:
        status = "=" * 119 + "\n"
        status += f"ViaBTC Miner | 算力: {hashrate:.2f} H/s | 礦工: {current_worker_name} | "
        status += f"已接受: {shares} | 已拒絕: {rejected_shares} "
        status += f"({(rejected_shares/(shares+1e-6)*100):.1f}%) | 區塊: {height}\n"
        status += "=" * 119
        print(status)
        
        # 添加詳細日誌
        if shares > 0:
            logg(f"[Stats] Hashrate: {hashrate:.2f} H/s")
            logg(f"[Stats] Accepted: {shares}")
            logg(f"[Stats] Rejected: {rejected_shares}")
            logg(f"[Stats] Reject Rate: {(rejected_shares/(shares+1e-6)*100):.1f}%")
            
    except Exception as e:
        log_error(f"Status display error: {str(e)}")

def print_mining_info(message, message_type="info"):
    """打印挖礦信息"""
    colors = {
        "info": Fore.WHITE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED
    }
    color = colors.get(message_type, Fore.WHITE)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Fore.BLUE}[{timestamp}]{color} {message}{Style.RESET_ALL}")

def parallel_hash_calculation(blockheader, nonce_start, batch_size=1000):
    """使用多進程並行哈希計算"""
    try:
        # 將批量分成多個子批次
        num_processes = cpu_count()
        sub_batch_size = batch_size // num_processes
        
        with Pool(processes=num_processes) as pool:
            # 創建子任務
            tasks = []
            for i in range(num_processes):
                start = nonce_start + (i * sub_batch_size)
                tasks.append((blockheader, start, sub_batch_size))
            
            # 並行執行
            results = pool.starmap(single_batch_calculation, tasks)
            
            # 合併結果
            all_results = []
            for batch_results, _ in results:
                all_results.extend(batch_results)
                if len(all_results) >= 2:
                    return all_results[:2], nonce_start + batch_size
            
            return all_results, nonce_start + batch_size
            
    except Exception as e:
        logg(f"Parallel hash error: {str(e)}")
        return [], nonce_start + batch_size

def generate_similar_hash(target_hash):
    """生成相似的哈希值"""
    try:
        base = int(target_hash, 16)
        # 小幅度調整以保持相似性
        adjustment = random.randint(-1000, 1000)
        new_hash = hex(base + adjustment)[2:].zfill(64)
        return new_hash
    except Exception as e:
        log_error(f"Hash generation error: {str(e)}")
        return target_hash

def handle_submit_error(error_code, error):
    """處理提交錯誤"""
    if error_code == 20:  # 無效的 share
        logg(f"[!] Invalid share: {error[1] if len(error) > 1 else 'unknown error'}")
        time.sleep(1)
    elif error_code == 21:  # 重複的 share
        logg("[!] Duplicate share")
        time.sleep(0.5)
    elif error_code == 10:  # 頻率限制
        logg("[!] Rate limit exceeded")
        time.sleep(2)
    else:
        logg(f"[!] Share rejected: {error}")
        time.sleep(1)

def monitor_mining_performance():
    """監控挖礦性能"""
    global total_shares, rejected_shares
    
    try:
        # 計算關鍵指標
        accept_rate = (total_shares / (total_shares + rejected_shares)) * 100 if (total_shares + rejected_shares) > 0 else 0
        
        # 如果接受率過低，調整策略
        if accept_rate < 90:
            logg(f"警告：接受率過低 ({accept_rate:.2f}%)")
            # 可以在這裡實現自動調整策略
            
        # 定期記錄性能數據
        if total_shares % 100 == 0:
            logg(f"性能報告：接受率 {accept_rate:.2f}% | 總計：{total_shares + rejected_shares}")
            
    except Exception as e:
        log_error(f"Performance monitoring error: {str(e)}")

def optimize_share_submission(shares, ctx, connection):
    """優化 share 提交策略"""
    try:
        # 控制提交頻率
        min_submit_interval = 1.0  # 最小提交間隔（秒）
        last_submit_time = time.time()  # 修正：使用 time.time() 初始化
        
        for nonce, hash_hex in shares:
            current_time = time.time()
            
            # 確保提交間隔
            if current_time - last_submit_time < min_submit_interval:
                time.sleep(min_submit_interval - (current_time - last_submit_time))
            
            # 驗證 share 品質
            if is_share_valid(hash_hex, ctx.target):
                # 提交 share
                submit_single_share(nonce, ctx, connection)
                last_submit_time = time.time()  # 修正：直接使用 time.time()
            
    except Exception as e:
        log_error(f"Share submission error: {str(e)}")

def is_share_valid(hash_hex, target):
    """驗證 share 品質"""
    try:
        hash_int = int(hash_hex, 16)
        target_int = int(target)
        
        # 確保 hash 值小於目標難度
        return hash_int <= target_int
        
    except Exception as e:
        log_error(f"Share validation error: {str(e)}")
        return False

def submit_single_share(nonce, ctx, connection):
    """提交單個 share"""
    try:
        # 構建提交數據
        payload = {
            "id": 4,
            "method": "mining.submit",
            "params": [
                current_worker_name,
                ctx.job_id,
                ctx.extranonce2,
                ctx.ntime,
                hex(nonce)[2:].zfill(8)
            ]
        }
        
        # 發送數據
        connection.sendall(json.dumps(payload).encode() + b'\n')
        
        # 等待響應
        response = connection.recv(4096).decode().strip()
        if response:
            response_data = json.loads(response)
            if response_data.get('result') is True:
                total_shares += 1
                logg(f"[*] Share accepted ({total_shares})")
                return True
            else:
                error = response_data.get('error')
                if error:
                    rejected_shares += 1
                    handle_submit_error(error[0] if isinstance(error, list) else None, error)
        
        return False
        
    except Exception as e:
        log_error(f"Share submission error: {str(e)}")
        return False

if __name__ == '__main__' :
    signal(SIGINT , handler)
    StartMining()
