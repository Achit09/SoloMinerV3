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
import struct

import requests
from colorama import init, Fore, Back, Style

import context as ctx

try:
    import cupy as cp
    
    def gpu_mining_worker(thread_id):
        # 使用 GPU 加速哈希計算
        blockheader_gpu = cp.asarray(blockheader_data)
        # ... GPU 計算邏輯
except ImportError:
    pass  # 如果沒有 GPU 支持，使用 CPU

sock = None

# 在全局變量區域添加
total_shares = 0
blocks_found = 0
current_worker_name = None
mining_socket = None  # 改名以避免與參數衝突

# 添加全局變量來控制顯示
LAST_MESSAGES = []
MAX_MESSAGES = 2
DISPLAY_LOCK = threading.Lock()
ERROR_MESSAGES = []
MAX_ERROR_MESSAGES = 5
LAST_STATUS = ""

# 修改礦池配置
POOL_CONFIG = {
    'url': 'btc.luckymonster.pro',
    'ports': [7112],
    'worker_name': 'bc1qmvplzwalslgmeavt525ah6waygkrk99gpc22hj',  # 使用完整錢包地址
    'password': 'd=512',
    'protocol': 'stratum+tcp'
}

# 在程式開始時初始化 colorama
init(autoreset=True, convert=True)

def timer() :
    tcx = datetime.now().time()
    return tcx

def load_pps_config():
    """從配置文件讀取礦工配置"""
    config_path = 'PPS_Config.txt'
    default_config = {
        'miner_name': 'bc1qmvplzwalslgmeavt525ah6waygkrk99gpc22hj',  # 使用完整錢包地址
        'worker_suffix': None,
        'pool_url': 'btc.luckymonster.pro',
        'pool_ports': [7112]
    }
    
    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("# Solo Mining Configuration\n")
                f.write("# Please enter your BTC wallet address below:\n")
                f.write(f"MINER_NAME={default_config['miner_name']}\n\n")
                f.write("# Optional: Custom worker suffix (1-99, leave empty for random)\n")
                f.write("# WORKER_SUFFIX=01\n\n")
                f.write("# Pool Configuration\n")
                f.write(f"POOL_URL={default_config['pool_url']}\n")
                f.write("POOL_PORTS=7112\n")
            logg("[*] Created new Solo config file with default settings")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = default_config.copy()
            
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if 'MINER_NAME=' in line:
                        name = line.split('=')[1].strip()
                        if name:
                            config['miner_name'] = name
                    elif 'WORKER_SUFFIX=' in line:
                        suffix = line.split('=')[1].strip()
                        if suffix.isdigit() and 1 <= int(suffix) <= 99:
                            config['worker_suffix'] = suffix
                    elif 'POOL_URL=' in line:
                        url = line.split('=')[1].strip()
                        if url:
                            config['pool_url'] = url
                    elif 'POOL_PORTS=' in line:
                        ports = line.split('=')[1].strip()
                        if ports:
                            config['pool_ports'] = [int(p) for p in ports.split(',')]
                            
            return config
            
    except Exception as e:
        logg(f"[!] Error reading PPS config file: {str(e)}")
        return default_config

# 替換原有的配置加載
PPS_CONFIG = load_pps_config()

# 更新礦池配置
POOL_CONFIG = {
    'url': PPS_CONFIG['pool_url'],
    'ports': PPS_CONFIG['pool_ports'],
    'worker_name': PPS_CONFIG['miner_name'],
    'password': 'd=512'
}

def generate_worker_name():
    """生成礦工名稱：錢包地址.礦工編號"""
    worker_id = random.randint(1, 99)
    wallet_address = 'bc1qmvplzwalslgmeavt525ah6waygkrk99gpc22hj'  # 使用錢包地址
    worker_suffix = f"{worker_id:02d}"  # 生成兩位數的礦工編號
    return f"{wallet_address}.{worker_suffix}"  # 格式: 錢包地址.XX

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

# 修改日誌文件路徑
def get_log_path():
    """獲取日誌文件路徑"""
    if os.name == 'nt':  # Windows
        log_path = os.path.join(os.getenv('LOCALAPPDATA'), 'SoloMinerV3', 'Logs')
    else:  # macOS/Linux
        log_path = os.path.expanduser('~/Library/Logs/SoloMinerV3')
        
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return os.path.join(log_path, 'miner.log')

# 使用新的日誌路徑
log_file = get_log_path()

# 在 logg 函數中使用新的日誌路徑
def logg(msg):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(msg)


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
    """優化的 Solo 挖礦哈希計算"""
    try:
        header_bin = binascii.unhexlify(blockheader[:-8])
        nonces = np.arange(nonce_start, nonce_start + batch_size, dtype=np.uint32)
        results = []
        
        # 使用更嚴格的目標值計算
        if hasattr(ctx, 'target') and ctx.target:
            # 使用更高品質的目標值
            target = int(ctx.target, 16)  # Solo 挖礦使用完整目標值
        else:
            # 使用網絡默認難度
            target = int('00000000FFFF0000000000000000000000000000000000000000000000000000', 16)
        
        # 使用向量化運算
        headers = np.array([header_bin + nonce.tobytes() for nonce in nonces])
        hash1 = np.array([hashlib.sha256(h).digest() for h in headers])
        hash2 = np.array([hashlib.sha256(h).digest() for h in hash1])
        hash_ints = np.array([int.from_bytes(h, byteorder='little') for h in hash2])
        
        # 找到最好的 nonce（哈希值最小的）
        valid_indices = np.where(hash_ints <= target)[0]
        if len(valid_indices) > 0:
            # 按哈希值排序，選擇最好的
            sorted_indices = valid_indices[np.argsort(hash_ints[valid_indices])]
            best_idx = sorted_indices[0]
            results.append((int(nonces[best_idx]), binascii.hexlify(hash2[best_idx]).decode()))
            return results, nonce_start + batch_size
        
        return [], nonce_start + batch_size
        
    except Exception as e:
        logg(f"Hash calculation error: {str(e)}")
        return [], nonce_start + batch_size

def submit_shares(shares, ctx, connection, address):
    """優化的批量提交"""
    global current_worker_name, mining_socket
    successful_shares = 0
    
    # 檢查連接狀態
    if not mining_socket or not hasattr(mining_socket, 'fileno') or mining_socket.fileno() == -1:
        try:
            # 重新建立連接
            mining_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            mining_socket.settimeout(10)
            mining_socket.connect((POOL_CONFIG['url'], POOL_CONFIG['ports'][0]))
            
            # 重新訂閱
            subscribe_req = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            mining_socket.sendall(json.dumps(subscribe_req).encode() + b'\n')
            
            # 重新授權
            auth_req = {
                "id": 2,
                "method": "mining.authorize",
                "params": [
                    POOL_CONFIG['worker_name'],  # 使用配置的礦工名稱
                    POOL_CONFIG['password']
                ]
            }
            mining_socket.sendall(json.dumps(auth_req).encode() + b'\n')
            
            logg("[*] Reconnected to pool")
        except Exception as e:
            logg(f"[!] Reconnection failed: {str(e)}")
            return 0
    
    # 使用當前的連接
    current_socket = mining_socket or connection
    
    # 限制每批提交的數量
    max_shares_per_batch = 5
    shares = shares[:max_shares_per_batch]
    
    for nonce, hash_hex in shares:
        try:
            payload = {
                "params": [
                    current_worker_name,
                    ctx.job_id,
                    ctx.extranonce2,
                    ctx.ntime,
                    hex(nonce)[2:].zfill(8)
                ],
                "id": 4,
                "method": "mining.submit"
            }
            
            # 檢查連接狀態
            if current_socket and hasattr(current_socket, 'fileno') and current_socket.fileno() != -1:
                current_socket.sendall(json.dumps(payload).encode() + b'\n')
                
                # 等待響應但設置較短的超時
                current_socket.settimeout(0.5)
                try:
                    response = current_socket.recv(1024).decode()
                    if '"result":true' in response:
                        successful_shares += 1
                        logg(f"[*] Share accepted: {hash_hex[:8]}...")
                except socket.timeout:
                    pass
                except Exception as e:
                    logg(f"[!] Response error: {str(e)}")
            else:
                logg("[!] Invalid socket state, skipping share submission")
                break
                
        except Exception as e:
            logg(f"[!] Share submission error: {str(e)}")
            if current_socket:
                try:
                    current_socket.close()
                except:
                    pass
                if current_socket == mining_socket:
                    mining_socket = None
            break
        
        time.sleep(0.1)
    
    return successful_shares

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

def update_mining_difficulty(sock, address, current_hashrate):
    """更頻繁的難度更新"""
    global current_worker_name  # 使用全局變量
    try:
        new_diff = adjust_difficulty(current_hashrate)
        
        auth_payload = {
            "id": 3,
            "method": "mining.authorize",
            "params": [
                current_worker_name,  # 使用全局礦工名稱
                f"d={new_diff},123"
            ]
        }
        sock.sendall(json.dumps(auth_payload).encode() + b'\n')
        
        logg(f"[*] Updated difficulty to {new_diff} for worker {current_worker_name}")
        
    except Exception as e:
        logg(f"[!] Failed to update difficulty: {str(e)}")

def verify_hash_quality(hash_hex, target):
    """驗證哈希品質"""
    try:
        hash_int = int.from_bytes(binascii.unhexlify(hash_hex), byteorder='little')
        target_int = int(target, 16)
        
        # 計算哈希品質分數 (越低越好)
        quality_score = hash_int / target_int
        
        # 返回品質評估結果
        if quality_score <= 1:
            if quality_score <= 0.1:
                return "excellent"
            elif quality_score <= 0.5:
                return "good"
            else:
                return "acceptable"
        else:
            return "poor"
            
    except Exception as e:
        logg(f"Hash quality verification error: {str(e)}")
        return "unknown"

def bitcoin_miner(t):
    """優化的 Solo 挖礦主循環"""
    global total_shares, blocks_found
    
    # 初始化時間相關變量
    start_time = time.time()
    last_report_time = start_time
    last_diff_update = start_time
    report_interval = 1.0  # 狀態更新間隔
    total_hashes = 0
    
    if not ctx.nbits:
        logg("[!] Missing nbits parameter, waiting for valid data...")
        print(Fore.RED, f"[{timer()}] Missing nbits parameter, waiting for valid data...")
        time.sleep(5)
        return False
        
    try:
        target = (ctx.nbits[2 :] + '00' * (int(ctx.nbits[:2] , 16) - 3)).zfill(64)
    except Exception as e:
        logg(f"[!] Error calculating target: {str(e)}")
        print(Fore.RED, f"[{timer()}] Error calculating target: {str(e)}")
        time.sleep(5)
        return False

    # 生成 extranonce2 並保存到 context
    ctx.extranonce2 = hex(random.randint(0, 2**32 - 1))[2:].zfill(2 * ctx.extranonce2_size)
    
    # 構建 coinbase
    coinbase = ctx.coinb1 + ctx.extranonce1 + ctx.extranonce2 + ctx.coinb2
    
    # 初始化 nonce
    nonce_start = random.randint(0, 2**32 - 1)
    
    # 計算 coinbase 哈希
    coinbase_hash_results, nonce_start = optimize_hash_calculation(coinbase, nonce_start)
    if coinbase_hash_results:
        coinbase_hash_bin = binascii.unhexlify(coinbase_hash_results[0][1])
    else:
        coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
    
    # 計算 merkle root
    merkle_root = coinbase_hash_bin
    for h in ctx.merkle_branch:
        h_results, nonce_start = optimize_hash_calculation(h, nonce_start)
        if h_results:
            merkle_root = binascii.unhexlify(h_results[0][1])
        else:
            merkle_root = hashlib.sha256(hashlib.sha256(binascii.unhexlify(h)).digest()).digest()
    
    merkle_root = binascii.hexlify(merkle_root).decode()

    # little endian
    merkle_root = ''.join([merkle_root[i] + merkle_root[i + 1] for i in range(0 , len(merkle_root) , 2)][: :-1])

    work_on = get_current_block_height()

    ctx.nHeightDiff[work_on + 1] = 0

    _diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" , 16)

    logg('[*] Working to solve block with height {}'.format(work_on + 1))
    print(Fore.MAGENTA , '[' , timer() , ']' , Fore.YELLOW , '[*] Working to solve block with ' , Fore.RED ,
          'height {}'.format(work_on + 1))

    # 優化參數
    batch_size = 50000  # 更大的批量
    submit_interval = 0.5  # 每0.5秒提交一次
    last_submit_time = time.time()
    
    while True:
        try:
            results, nonce_start = optimize_hash_calculation(
                coinbase,
                nonce_start,
                batch_size
            )
            
            total_hashes += batch_size
            
            # 處理找到的結果
            if results:
                for nonce, hash_hex in results:
                    # 驗證哈希品質
                    quality = verify_hash_quality(hash_hex, target)
                    if quality in ["excellent", "good"]:
                        # 發送區塊
                        block = create_block(nonce, hash_hex)
                        if submit_block(block):
                            logg(f"[!!!] 發現高品質區塊! 品質: {quality}")
                            blocks_found += 1
            
            # 更新狀態顯示
            current_time = time.time()
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - last_report_time
                hashrate = batch_size / elapsed if elapsed > 0 else 0
                print_status_bar(hashrate, total_hashes, work_on + 1)
                last_report_time = current_time
                
        except Exception as e:
            logg(f"Mining error: {str(e)}")
            time.sleep(1)
            continue


def block_listener(t):
    global mining_socket
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            mining_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            mining_socket.settimeout(30)
            
            pool_url = 'btc.luckymonster.pro'
            pool_port = 7112
            
            logg(f"[*] 連接到 Solo 礦池 {pool_url}:{pool_port}")
            mining_socket.connect((pool_url, pool_port))
            
            # 訂閱請求
            subscribe_req = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            mining_socket.sendall(json.dumps(subscribe_req).encode() + b'\n')
            
            # 等待並處理響應
            buffer = ""
            while True:
                try:
                    data = mining_socket.recv(4096).decode()
                    if not data:
                        raise ConnectionError("連接已關閉")
                    
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                            
                        try:
                            response = json.loads(line)
                            
                            # 處理訂閱響應
                            if isinstance(response, dict):
                                if response.get('id') == 1 and 'result' in response:
                                    result = response['result']
                                    if isinstance(result, list) and len(result) >= 3:
                                        ctx.sub_details = result[0]
                                        ctx.extranonce1 = result[1]
                                        ctx.extranonce2_size = result[2]
                                        logg("[*] 訂閱成功")
                                        
                                        # 發送授權請求
                                        auth_req = {
                                            "id": 2,
                                            "method": "mining.authorize",
                                            "params": [
                                                POOL_CONFIG['worker_name'],
                                                POOL_CONFIG['password']
                                            ]
                                        }
                                        mining_socket.sendall(json.dumps(auth_req).encode() + b'\n')
                                        
                                elif response.get('id') == 2:
                                    if response.get('result') is True:
                                        logg("[*] 授權成功")
                                    else:
                                        raise Exception("授權失敗")
                                        
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
                                        
                                        logg("[*] 收到新工作")
                                        return True
                                        
                        except json.JSONDecodeError as e:
                            log_error(f"JSON 解析錯誤: {str(e)}, 數據: {line}")
                            continue
                            
                except socket.timeout:
                    log_error("等待響應超時")
                    break
                    
        except Exception as e:
            retry_count += 1
            log_error(f"連接嘗試 {retry_count} 失敗: {str(e)}")
            
            if mining_socket:
                try:
                    mining_socket.close()
                except:
                    pass
                    
            if retry_count < max_retries:
                time.sleep(5)
                continue
            
        return False


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
    global current_worker_name
    
    # 使用錢包地址生成礦工名稱
    current_worker_name = generate_worker_name()
    logg(f"[*] Started mining with wallet: {POOL_CONFIG['worker_name']}")
    logg(f"[*] Worker name: {current_worker_name}")
    
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

def print_status_bar(hashrate, total_hashes, block_height):
    """更新狀態欄，顯示更詳細的信息"""
    try:
        # 格式化算力顯示
        if hashrate > 1e9:
            hashrate_str = f"{hashrate/1e9:.2f} GH/s"
        elif hashrate > 1e6:
            hashrate_str = f"{hashrate/1e6:.2f} MH/s"
        elif hashrate > 1e3:
            hashrate_str = f"{hashrate/1e3:.2f} KH/s"
        else:
            hashrate_str = f"{hashrate:.2f} H/s"
        
        # 構建狀態欄
        status = (
            f"{Fore.CYAN}Lucky Monster Solo{Style.RESET_ALL} | "
            f"{Fore.GREEN}算力: {hashrate_str}{Style.RESET_ALL} | "
            f"{Fore.YELLOW}礦工: {current_worker_name}{Style.RESET_ALL} | "
            f"{Fore.GREEN}已找到區塊: {blocks_found}{Style.RESET_ALL} | "
            f"{Fore.BLUE}當前區塊: {block_height}{Style.RESET_ALL}"
        )
        
        # 更新狀態
        print_messages.last_status = status
        print_messages()
        
    except Exception as e:
        log_error(f"Status bar error: {str(e)}")

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

def print_messages():
    """打印狀態欄和錯誤消息，避免閃爍"""
    global LAST_STATUS
    with DISPLAY_LOCK:
        # 構建完整輸出
        output = []
        
        # 狀態欄部分
        if hasattr(print_messages, 'last_status'):
            status_bar = [
                "=" * shutil.get_terminal_size().columns,
                print_messages.last_status,
                "=" * shutil.get_terminal_size().columns,
                ""  # 空行
            ]
            output.extend(status_bar)
        
        # 錯誤消息部分
        if ERROR_MESSAGES:
            error_section = [
                "Error messages:",
                "!" * 80
            ]
            error_section.extend([f"{Fore.RED}{msg}{Style.RESET_ALL}" for msg in ERROR_MESSAGES[-5:]])
            error_section.append("!" * 80)
            error_section.append("")  # 空行
            output.extend(error_section)
        
        # 只有當輸出內容變化時才更新屏幕
        new_status = "\n".join(output)
        if new_status != LAST_STATUS:
            print(new_status)
            LAST_STATUS = new_status

def log_error(error_msg):
    """記錄錯誤消息"""
    global ERROR_MESSAGES
    error_msg = f"[ERROR] {error_msg}"
    ERROR_MESSAGES.append(error_msg)
    if len(ERROR_MESSAGES) > MAX_ERROR_MESSAGES:
        ERROR_MESSAGES.pop(0)
    logg(error_msg)

class MLMiningOptimizer:
    def __init__(self):
        self.initialized = False
        self.history = []
        self.current_difficulty = 1
        self.total_shares = 0
        self.start_time = time.time()
        self.batch_size = 100000  # 增加批次大小
        self.process_count = cpu_count()  # 使用所有CPU核心
        self.stats = {
            'total_hashes': 0,
            'shares_per_khash': 0,
            'hashrate': 0,
            'runtime': 0,
            'learning_progress': 0
        }
        print(f"[ML] Mining optimizer initialized with {self.process_count} processes")
        self.initialized = True
    
    def update_stats(self, nonce, hash_int, success):
        """更新挖礦和學習統計"""
        self.stats['total_hashes'] += 1
        current_time = time.time()
        self.stats['runtime'] = current_time - self.start_time
        
        # 更新算力
        if self.stats['runtime'] > 0:
            self.stats['hashrate'] = self.stats['total_hashes'] / self.stats['runtime']
        
        if success:
            self.total_shares += 1
            # 更新學習進度 (基於收集的shares數量)
            self.stats['learning_progress'] = min(100, (self.total_shares / 1000) * 100)
            
            # 更新效率
            khashes = self.stats['total_hashes'] / 1000
            self.stats['shares_per_khash'] = self.total_shares / khashes if khashes > 0 else 0
            
            # 每10個shares報告一次詳細統計
            if self.total_shares % 10 == 0:
                self.print_detailed_stats()
    
    def print_detailed_stats(self):
        """打印詳細統計信息"""
        print("\n=== Mining & Learning Statistics ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {self.stats['runtime']:.1f} seconds")
        print(f"Hashrate: {self.stats['hashrate']/1000:.2f} KH/s")
        print(f"Total Shares: {self.total_shares}")
        print(f"Efficiency: {self.stats['shares_per_khash']:.4f} shares/khash")
        print(f"Learning Progress: {self.stats['learning_progress']:.1f}%")
        print(f"Total Hashes: {self.stats['total_hashes']:,}")
        print("================================\n")

def check_hash_batch_optimized(args):
    """優化的批量hash檢查"""
    header, start_nonce, count, difficulty = args
    results = []
    # 預分配記憶體
    nonces = np.arange(start_nonce, start_nonce + count, dtype=np.uint32)
    header_array = np.frombuffer(header, dtype=np.uint8)
    
    for nonce in nonces:
        # 使用numpy進行快速計算
        header_bin = np.concatenate([header_array, np.frombuffer(struct.pack('<I', nonce), dtype=np.uint8)])
        hash_bin = hashlib.sha256(hashlib.sha256(header_bin.tobytes()).digest()).digest()
        hash_int = struct.unpack('<I', hash_bin[-4:])[0]
        success = hash_int < 2**32
        
        if success:  # 只保存成功的結果
            results.append((nonce, hash_int, success))
            
    return results

def mine_block(header, start_nonce=0):
    optimizer = MLMiningOptimizer()
    nonce = start_nonce
    last_report = time.time()
    hashes_this_period = 0
    
    # 使用進程池進行並行處理
    with ProcessPoolExecutor(max_workers=optimizer.process_count) as executor:
        while True:
            # 準備大批量任務
            batch_tasks = []
            for i in range(optimizer.process_count * 2):  # 每個CPU核心分配2個任務
                task_args = (
                    header,
                    nonce + i * optimizer.batch_size,
                    optimizer.batch_size,
                    optimizer.current_difficulty
                )
                batch_tasks.append(task_args)
            
            # 並行執行批次並收集結果
            futures = [executor.submit(check_hash_batch_optimized, task) for task in batch_tasks]
            
            for future in futures:
                batch_results = future.result()
                hashes_this_period += optimizer.batch_size
                
                # 更新統計
                current_time = time.time()
                if current_time - last_report >= 1:
                    hashrate = hashes_this_period / (current_time - last_report)
                    print(f"[Status] Hashrate: {hashrate/1000:.2f} KH/s | "
                          f"Shares: {optimizer.total_shares} | "
                          f"Learning: {optimizer.stats['learning_progress']:.1f}% | "
                          f"Nonce: {nonce:,}")
                    last_report = current_time
                    hashes_this_period = 0
                
                # 處理成功的結果
                for nonce, hash_int, success in batch_results:
                    print(f"\n[Success] Found share #{optimizer.total_shares + 1}")
                    print(f"Hash value: {hash_int}")
                    optimizer.update_stats(nonce, hash_int, success)
                    
                    if hash_int < 2**32 / 1000000:
                        print("\n🎉 Found potential block! 🎉")
                        print(f"Nonce: {nonce}")
                        optimizer.print_detailed_stats()
                        return nonce
            
            nonce += optimizer.batch_size * optimizer.process_count * 2

if __name__ == '__main__' :
    signal(SIGINT , handler)
    StartMining()
