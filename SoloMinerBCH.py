import hashlib
import json
import socket
import struct
import threading
import time
import traceback
from datetime import datetime
import multiprocessing
from multiprocessing import cpu_count, Pool, freeze_support
import numpy as np
import os
import signal
import math

# 日誌設置
def logg(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_error(error_msg):
    logg(f"ERROR: {error_msg}")

class BCHMiningOptimizer:
    def __init__(self):
        self.initialized = False
        self.history = []
        self.current_difficulty = 1
        self.total_shares = 0
        self.start_time = time.time()
        self.batch_size = 2000000  # 增加到200萬
        self.process_count = 1
        self.stats = {
            'total_hashes': 0,
            'shares_per_khash': 0,
            'hashrate': 0,
            'runtime': 0,
            'learning_progress': 0
        }
        self.target_difficulty = int(2**240)
        self.share_difficulty = self.target_difficulty
        self.min_share_difficulty = 1_000_000_000
        self.total_hashes = 0
        self.last_update = time.time()
        self.hashes_this_period = 0
        self.mining_active = False
        self.simulation_factor = 200000  # 提高到20萬倍
        self.shares_found = 0
        self.last_share_time = time.time()
        self.share_interval = 5  # 縮短目標間隔
        self.difficulty_adjustment = 1.0
        self.hash_multiplier = 5000  # 提高到5000倍
        self.current_hashrate = 0
        logg("[*] BCH Mining optimizer initialized")
        self.initialized = True
    
    def set_target_difficulty(self, target):
        """設置新的目標難度"""
        if target is None or target <= 0:
            log_error("Invalid target difficulty, using default")
            target = int(2**240)
        
        target = int(target)
        self.target_difficulty = target
        
        # 確保share難度不低於最小值
        min_difficulty = self.min_share_difficulty
        calculated_share_diff = target // 100000
        
        self.share_difficulty = max(calculated_share_diff, min_difficulty)
        self.current_difficulty = self.share_difficulty
        
        logg(f"[ML] 目標: {format_difficulty(target)}")
        logg(f"[ML] Share: {format_difficulty(self.share_difficulty)}")
    
    def update_stats(self, nonce, hash_int, success):
        self.stats['total_hashes'] += 1
        current_time = time.time()
        self.stats['runtime'] = current_time - self.start_time
        
        if self.stats['runtime'] > 0:
            self.stats['hashrate'] = self.stats['total_hashes'] / self.stats['runtime']
        
        if success:
            self.total_shares += 1
            self.stats['learning_progress'] = min(100, (self.total_shares / 1000) * 100)
            khashes = self.stats['total_hashes'] / 1000
            self.stats['shares_per_khash'] = self.total_shares / khashes if khashes > 0 else 0
            
            if self.total_shares % 10 == 0:
                self.print_detailed_stats()
    
    def print_detailed_stats(self):
        print("\n=== BCH Mining Statistics ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {self.stats['runtime']:.1f} seconds")
        print(f"Hashrate: {self.stats['hashrate']/1000:.2f} KH/s")
        print(f"Total Shares: {self.total_shares}")
        print(f"Efficiency: {self.stats['shares_per_khash']:.4f} shares/khash")
        print(f"Learning Progress: {self.stats['learning_progress']:.1f}%")
        print(f"Total Hashes: {self.stats['total_hashes']:,}")
        print("================================\n")

    def update_hashrate(self, new_hashes):
        """更新哈希率和總哈希數"""
        current_time = time.time()
        time_diff = current_time - self.last_update
        
        # 更新總哈希數
        self.total_hashes = new_hashes
        
        if time_diff >= 0.5:  # 每0.5秒更新一次
            # 計算這個時間段的哈希率
            hashes_diff = new_hashes - self.hashes_this_period
            current_rate = hashes_diff / time_diff if time_diff > 0 else 0
            
            # 平滑處理
            if self.current_hashrate == 0:
                self.current_hashrate = current_rate
            else:
                self.current_hashrate = (self.current_hashrate * 0.7) + (current_rate * 0.3)
            
            self.last_update = current_time
            self.hashes_this_period = new_hashes
            self.stats['hashrate'] = self.current_hashrate
    
    @property
    def hashrate(self):
        """獲取當前哈希率"""
        return self.stats['hashrate']
    
    @property
    def total_hash_count(self):
        """獲取總哈希數"""
        return self.total_hashes

class BCHSoloMiner:
    def __init__(self, address):
        self.server = ('bch.luckymonster.pro', 6112)
        self.address = address
        self.sock = None
        self.working = True
        self.job = None
        self.optimizer = BCHMiningOptimizer()  # 只在這裡初始化一次
        self.current_height = 0
        self.submitted_shares = set()
        self.last_status_update = time.time()
        self.current_mining_thread = None
        self.mining_lock = threading.Lock()
        self.processed_hashes = 0  # 添加總哈希數計數器
    
    def connect(self):
        while self.working:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server)
                self.subscribe()
                self.authorize()
                return True
            except Exception as e:
                log_error(f"Connection error: {str(e)}")
                time.sleep(5)
        return False
    
    def subscribe(self):
        subscribe = {
            'id': 1,
            'method': 'mining.subscribe',
            'params': []
        }
        self.send_message(subscribe)
        response = self.recv_message()
        logg("[*] 訂閱成功")
    
    def authorize(self):
        authorize = {
            'id': 2,
            'method': 'mining.authorize',
            'params': [self.address, 'x']
        }
        self.send_message(authorize)
        response = self.recv_message()
        logg("[*] 授權成功")
    
    def send_message(self, msg):
        try:
            data = json.dumps(msg) + '\n'
            self.sock.send(data.encode())
        except Exception as e:
            log_error(f"Send error: {str(e)}")
            self.reconnect()
    
    def recv_message(self):
        try:
            data = self.sock.recv(4096)
            messages = data.decode().split('\n')
            for msg in messages:
                if msg:
                    return json.loads(msg)
        except Exception as e:
            log_error(f"Receive error: {str(e)}")
            self.reconnect()
        return None
    
    def reconnect(self):
        self.sock.close()
        time.sleep(5)
        self.connect()
    
    def process_job(self, job):
        if not job or 'params' not in job:
            return
            
        with self.mining_lock:
            # 停止當前挖礦
            self.working = False
            if self.current_mining_thread and self.current_mining_thread.is_alive():
                try:
                    self.current_mining_thread.join(timeout=2)
                except RuntimeError:
                    pass
            
            # 重置工作狀態，但保留哈希計數
            self.working = True
            self.submitted_shares.clear()
            
            job_id = job['params'][0]
            prevhash = bytes.fromhex(job['params'][1])
            coinb1 = bytes.fromhex(job['params'][2])
            coinb2 = bytes.fromhex(job['params'][3])
            merkle_branches = [bytes.fromhex(h) for h in job['params'][4]]
            version = struct.pack('<I', int(job['params'][5], 16))
            nbits = bytes.fromhex(job['params'][6])
            ntime = bytes.fromhex(job['params'][7])
            clean_jobs = job['params'][8]
            
            # 解析區塊高度的新方法
            try:
                self.debug_coinb1(coinb1)  # 添加這行來獲取更多信息
                # 在 coinb1 中尋找區塊高度標記（0x03 後面跟著3個字節的高度）
                height_marker = b'\x03'
                marker_pos = coinb1.find(height_marker)
                if marker_pos != -1:
                    # 讀取3字節的高度數據
                    height_bytes = coinb1[marker_pos+1:marker_pos+4]
                    self.current_height = int.from_bytes(height_bytes, 'little')
                    logg(f"[*] 當前區塊高度: {self.current_height}")
                else:
                    # 嘗試另一種格式（0x04 後面跟著4個字節的高度）
                    height_marker = b'\x04'
                    marker_pos = coinb1.find(height_marker)
                    if marker_pos != -1:
                        height_bytes = coinb1[marker_pos+1:marker_pos+5]
                        self.current_height = int.from_bytes(height_bytes, 'little')
                        logg(f"[*] 當前區塊高度: {self.current_height}")
                    else:
                        log_error("無法找到區塊高度標記")
            except Exception as e:
                log_error(f"解析區塊高度失敗: {str(e)}")
            
            # 生成 extranonce2 (4字節)
            extranonce2 = struct.pack('<I', int.from_bytes(os.urandom(4), 'little'))
            
            # 計算 merkle root
            coinbase = coinb1 + extranonce2 + coinb2
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle_root = coinbase_hash
            for branch in merkle_branches:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch).digest()).digest()
            
            # 計算目標難度
            nbits_int = int.from_bytes(nbits, 'little')
            exp = nbits_int >> 24
            mant = nbits_int & 0xffffff
            
            # 更精確的目標計算
            if mant > 0x7fffff:
                mant = 0x7fffff
            target = mant * (2 ** (8 * (exp - 3)))
            target = min(target, 2**256 - 1)
            
            # 使用 ASIC 標準的難度計算
            network_diff = 408_263_659_176
            share_diff = network_diff / 2000  # 使用更寬鬆的 share 難度
            max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
            share_target = int(max_target / share_diff)
            
            logg(f"[*] 收到新工作 - 區塊高度: {self.current_height}")
            logg(f"[*] 網絡難度: {format_difficulty_value(network_diff)}")
            logg(f"[*] Share難度: {format_difficulty_value(share_diff)}")
            
            self.job = {
                'id': job_id,
                'prevhash': prevhash,
                'coinb1': coinb1,
                'coinb2': coinb2,
                'merkle_branches': merkle_branches,
                'version': version,
                'nbits': nbits,
                'ntime': ntime,
                'merkle_root': merkle_root,
                'extranonce2': extranonce2.hex(),
                'target': max_target,
                'share_target': share_target,
                'network_diff': network_diff,
                'share_diff': share_diff
            }
            
            # 更新挖礦優化器的難度設置
            self.optimizer.set_target_difficulty(share_target)
            
            # 啟動新的挖礦線程
            self.current_mining_thread = threading.Thread(target=self.mine)
            self.current_mining_thread.daemon = True  # 設置為守護線程
            self.current_mining_thread.start()
    
    def mine(self):
        """優化的挖礦主循環"""
        if not self.job:
            return
        
        header_bin = b''.join([
            self.job['version'],
            self.job['prevhash'],
            self.job['merkle_root'],
            self.job['ntime'],
            self.job['nbits']
        ])
        
        nonce = int.from_bytes(os.urandom(4), 'little')  # 隨機起始點
        total_hashes = 0
        last_hash_update = time.time()
        
        while self.working and self.job:
            try:
                result, hashes = check_hash_batch_optimized((
                    header_bin,
                    nonce,
                    100000,  # 增加批次大小
                    0
                ))
                
                # 更新哈希計數
                total_hashes += hashes
                
                # 每秒更新一次哈希率
                current_time = time.time()
                if current_time - last_hash_update >= 0.5:
                    self.optimizer.update_hashrate(total_hashes)
                    last_hash_update = current_time
                
                if result:
                    self.submit_share(result[0][0])
                
                nonce = (nonce + 100000) & 0xFFFFFFFF
                time.sleep(0.001)
                
            except Exception as e:
                log_error(f"Mining error: {str(e)}")
                time.sleep(0.1)
    
    def submit_share(self, nonce):
        """提交 share 到礦池"""
        if not self.job:
            return
        
        try:
            submit = {
                'id': 4,
                'method': 'mining.submit',
                'params': [
                    self.address,
                    self.job['id'],
                    self.job['extranonce2'],
                    self.job['ntime'].hex(),
                    f'{nonce:08x}'
                ]
            }
            
            logg(f"[*] 提交 share - Job ID: {self.job['id']}, Nonce: {nonce:08x}")
            self.send_message(submit)
            
            # 等待響應
            start_time = time.time()
            while time.time() - start_time < 5:  # 最多等待5秒
                response = self.recv_message()
                if response and response.get('id') == 4:  # 確認是 submit 的響應
                    if response.get('result'):
                        logg("[*] ✅ Share 被接受!")
                        self.optimizer.shares_found += 1
                        return True
                    elif response.get('error'):
                        error = response['error']
                        log_error(f"❌ Share 被拒絕: {error}")
                        if isinstance(error, list) and len(error) > 1:
                            log_error(f"拒絕原因: {error[1]}")
                        return False
                time.sleep(0.1)
            
            log_error("Share 提交超時")
            return False
            
        except Exception as e:
            log_error(f"提交 share 時出錯: {str(e)}")
            traceback.print_exc()
            return False
    
    def verify_share(self, hash_int):
        """驗證 share 是否符合難度要求"""
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        share_target = int(max_target * self.optimizer.share_difficulty)
        
        if hash_int <= share_target:
            logg(f"[DEBUG] Share 驗證通過")
            logg(f"[DEBUG] Hash: {hash_int:064x}")
            logg(f"[DEBUG] Target: {share_target:064x}")
            return True
        return False
    
    def run(self):
        if not self.connect():
            return
            
        # 清空控制台並初始化
        print('\033[2J\033[H', end='')
        last_line = ''
        
        while self.working:
            try:
                current_time = time.time()
                # 更改為每0.5秒更新一次，使用覆蓋式輸出
                if current_time - self.last_status_update >= 0.5:
                    self.print_mining_stats_overlay()
                    self.last_status_update = current_time
                
                message = self.recv_message()
                if message:
                    if message.get('method') == 'mining.notify':
                        # 清除最後一行
                        print('\033[K', end='')
                        self.process_job(message)
                    elif message.get('error'):
                        # 清除最後一行並打印錯誤
                        print('\033[K', end='')
                        log_error(f"Server error: {message['error']}")
            except Exception as e:
                log_error(f"Error: {str(e)}")
                self.reconnect()

    def print_mining_stats_overlay(self):
        """優化的挖礦統計顯示"""
        current_time = time.time()
        runtime = current_time - self.optimizer.start_time
        shares_per_min = (self.optimizer.shares_found * 60) / runtime if runtime > 0 else 0
        
        # 使用優化器的實際數據
        status = (
            f"\r[BCH] "
            f"H:{self.current_height} | "
            f"Speed:{format_hashrate(self.optimizer.hashrate)}H/s | "
            f"S:{self.optimizer.shares_found}({shares_per_min:.1f}/m) | "
            f"Hash:{format_hashes(self.optimizer.total_hash_count)}"
        )
        
        print('\r\033[K' + status, end='', flush=True)

    def debug_coinb1(self, coinb1):
        """調試 coinb1 的內容 - 僅在調試模式下使用"""
        if os.getenv('DEBUG'):  # 只在設置 DEBUG 環境變量時輸出
            logg("=== Coinb1 Debug Info ===")
            logg(f"Raw hex: {coinb1.hex()}")
            logg(f"Length: {len(coinb1)} bytes")
            bytes_str = ' '.join(f'{b:02x}' for b in coinb1)
            logg(f"Bytes: {bytes_str}")
            
            for i in range(len(coinb1)-4):
                if coinb1[i] in [0x03, 0x04]:
                    logg(f"Possible height marker at position {i}: {coinb1[i]:02x}")
                    next_bytes = coinb1[i+1:i+5]
                    logg(f"Following bytes: {next_bytes.hex()}")

def calculate_target(nbits):
    """根據 nbits 計算目標值"""
    exp = nbits >> 24
    mant = nbits & 0xffffff
    if mant > 0x7fffff:
        mant = 0x7fffff
    target = mant * (2 ** (8 * (exp - 3)))
    return min(target, 2**256 - 1)

def calculate_difficulty(target):
    """計算難度值"""
    try:
        max_target = 2**256 - 1
        if target <= 0:
            return 0
        return max_target / target
    except:
        return 0

def check_hash_batch_optimized(args):
    """優化的挖礦算法"""
    header_bin, start_nonce, count, target = args
    results = []
    local_hashes = 0
    
    try:
        # 使用礦池設置的 share 難度
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        # 使用固定的 share 難度 (0.001)
        difficulty = 0.001
        target = int(max_target * difficulty)
        
        for nonce in range(start_nonce, start_nonce + count):
            header_with_nonce = header_bin + struct.pack('<I', nonce)
            hash_bin = hashlib.sha256(hashlib.sha256(header_with_nonce).digest()).digest()
            hash_int = int.from_bytes(hash_bin, 'big')
            local_hashes += 1
            
            if hash_int <= target:
                logg(f"[DEBUG] 找到 share - Hash: {hash_int:064x}")
                logg(f"[DEBUG] 目標值: {target:064x}")
                results.append((nonce, hash_int, True))
                return results, local_hashes
            
            if local_hashes >= 1000:  # 每批次處理1000個哈希
                break
                
    except Exception as e:
        log_error(f"Error in hash batch: {str(e)}")
        traceback.print_exc()
    
    return results, local_hashes

def init_worker():
    """初始化工作進程"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def mine_block(header_bin, start_nonce=0, optimizer=None):
    """優化的挖礦主函數"""
    nonce = start_nonce & 0xFFFFFFFF
    total_hashes = 0
    last_submit = time.time()
    min_submit_interval = 0.1  # 最小提交間隔100ms
    
    while True:
        try:
            result, hashes = check_hash_batch_optimized((
                header_bin,
                nonce,
                50000,  # 更大的批次
                0
            ))
            
            total_hashes += hashes
            if optimizer:
                optimizer.hashes_this_period = total_hashes
                optimizer.update_hashrate(total_hashes)
            
            if result:
                current_time = time.time()
                if current_time - last_submit >= min_submit_interval:
                    last_submit = current_time
                    return result[0][0]
            
            nonce = (nonce + 50000) & 0xFFFFFFFF
            time.sleep(0.0001)
            
        except Exception as e:
            log_error(f"Error in mining loop: {str(e)}")
            time.sleep(0.01)
    
    return None

def format_difficulty(diff):
    """格式化難度值，使用科學記數法"""
    return f"{diff:.2e}"

def format_hashrate(hashrate):
    """格式化哈希率顯示"""
    if hashrate >= 1e12:  # TH/s
        return f"{hashrate/1e12:.2f}T"
    elif hashrate >= 1e9:  # GH/s
        return f"{hashrate/1e9:.2f}G"
    elif hashrate >= 1e6:  # MH/s
        return f"{hashrate/1e6:.2f}M"
    elif hashrate >= 1e3:  # KH/s
        return f"{hashrate/1e3:.2f}K"
    else:
        return f"{hashrate:.2f}"

def format_hashes(hashes):
    """格式化哈希數顯示"""
    if hashes >= 1e12:  # T
        return f"{hashes/1e12:.1f}T"
    elif hashes >= 1e9:  # G
        return f"{hashes/1e9:.1f}G"
    elif hashes >= 1e6:  # M
        return f"{hashes/1e6:.1f}M"
    elif hashes >= 1e3:  # K
        return f"{hashes/1e3:.1f}K"
    else:
        return str(int(hashes))

def format_difficulty_value(diff):
    """格式化難度值顯示"""
    if diff >= 1e12:
        return f"{diff/1e12:.2f}T"
    elif diff >= 1e9:
        return f"{diff/1e9:.2f}G"
    elif diff >= 1e6:
        return f"{diff/1e6:.2f}M"
    elif diff >= 1e3:
        return f"{diff/1e3:.2f}K"
    else:
        return f"{diff:.2f}"

if __name__ == '__main__':
    # Windows多進程支持
    freeze_support()
    
    # 顯示系統資源建議
    logg("""
    === 系統要求 ===
    1. 建議物理內存：8GB 以上
    2. 建議虛擬內存：16GB 以上
    3. 可用硬碟空間：10GB 以上
    ===============
    """)
    
    # 設置進程啟動方法
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # BCH地址 (CashAddr格式)
    address = "qr52dusx0lkyevyjvate6yq87uhvcs8kl5r25ez882"
    
    logg(f"Starting BCH Solo Miner")
    logg(f"Mining Address: {address}")
    logg(f"Pool: bch.luckymonster.pro:6112")
    logg(f"Mode: SOLO")
    
    miner = BCHSoloMiner(address)
    try:
        miner.run()
    except KeyboardInterrupt:
        miner.working = False
        logg("Shutting down...")
    except Exception as e:
        log_error(f"Fatal error: {str(e)}")
        traceback.print_exc() 