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
import os
import signal
import random
import pyscrypt

# 日誌設置
def logg(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_error(error_msg):
    logg(f"ERROR: {error_msg}")

class DogeMiningOptimizer:
    def __init__(self):
        self.initialized = False
        self.total_shares = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.total_hashes = 0
        self._hashrate = 0
        self.shares_found = 0
        self.hashes_this_period = 0
        self.mining_active = False
        self.stats = {'hashrate': 0}
        logg("[*] DOGE Mining optimizer initialized")
        self.initialized = True
    
    @property
    def hashrate(self):
        return self._hashrate
    
    @hashrate.setter
    def hashrate(self, value):
        self._hashrate = value
    
    def update_hashrate(self, total_hashes):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self._hashrate = total_hashes / elapsed
    
    @property
    def total_hash_count(self):
        """獲取總哈希數"""
        return self.total_hashes

class DogeSoloMiner:
    def __init__(self, worker_name):
        # 修改連接設定為 Dogecoin 礦池
        self.server = ('doge.luckymonster.pro', 5112)
        self.username = worker_name  # 使用帶有編號的礦工名稱
        self.address = worker_name.split('.')[0]  # 保存原始地址
        self.worker_id = worker_name.split('.')[1]  # 保存礦工編號
        self.sock = None
        self.working = True
        self.job = None
        self.optimizer = DogeMiningOptimizer()
        self.current_height = 0
        self.submitted_shares = set()
        self.last_status_update = time.time()
        self.current_mining_thread = None
        self.mining_lock = threading.Lock()
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.network_difficulty = 57.322e6  # 57.322 M 網絡難度
        self.current_difficulty = 0.001  # 初始 share 難度
        self.message_queue = []
        self.message_lock = threading.Lock()
        self.network_thread = None
        self.mining_thread = None
        self.stats_thread = None
        logg(f"[*] 礦工名稱: {self.username} (ID: {self.worker_id})")
    
    def connect(self):
        """連接到礦池"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                
                # 創建新的 socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10)
                
                # 連接到伺服器
                logg(f"[*] 正在連接到 {self.server[0]}:{self.server[1]}...")
                self.sock.connect(self.server)
                
                # 訂閱
                subscribe = {
                    'id': 1,
                    'method': 'mining.subscribe',
                    'params': [
                        'dogeminer/2.0.0',
                        None
                    ]
                }
                
                if not self.send_message(subscribe):
                    raise ConnectionError("無法發送訂閱請求")
                
                # 等待訂閱響應
                response = None
                timeout = time.time() + 10
                
                while time.time() < timeout:
                    response = self.recv_message()
                    if response and response.get('id') == 1:  # 確認是訂閱響應
                        if response.get('error'):
                            raise ConnectionError(f"訂閱失敗: {response['error']}")
                        
                        # 解析訂閱響應
                        if response.get('result'):
                            subscription = response['result']
                            if isinstance(subscription, list) and len(subscription) >= 2:
                                self.extranonce1 = subscription[1]
                                self.extranonce2_size = subscription[2] if len(subscription) > 2 else 4
                                logg(f"[*] 訂閱成功 - ExtraNonce1: {self.extranonce1}")
                                break  # 成功獲取訂閱響應
                            else:
                                raise ConnectionError("無效的訂閱響應格式")
                    time.sleep(0.1)
                
                if not response or response.get('id') != 1:
                    raise ConnectionError("等待訂閱響應超時")
                
                # 授權
                authorize = {
                    'id': 2,
                    'method': 'mining.authorize',
                    'params': [
                        self.username,
                        "d=57322000"
                    ]
                }
                
                if not self.send_message(authorize):
                    raise ConnectionError("無法發送授權請求")
                
                # 等待授權響應和工作通知
                got_auth = False
                got_notify = False
                timeout = time.time() + 10
                
                while time.time() < timeout and not (got_auth and got_notify):
                    message = self.recv_message()
                    if not message:
                        time.sleep(0.1)
                        continue
                    
                    if message.get('id') == 2:  # 授權響應
                        if message.get('result') is True:
                            got_auth = True
                            logg("[*] 授權成功")
                        elif message.get('error'):
                            raise ConnectionError(f"授權失敗: {message['error']}")
                    
                    elif message.get('method') == 'mining.notify':  # 工作通知
                        self.process_job(message)
                        got_notify = True
                        logg("[*] 收到工作通知")
                    
                    elif message.get('method') == 'mining.set_difficulty':  # 難度設置
                        self.current_difficulty = message['params'][0]
                        logg(f"[*] 設置難度: {self.current_difficulty}")
                
                if not got_auth:
                    raise ConnectionError("未收到授權響應")
                if not got_notify:
                    raise ConnectionError("未收到工作通知")
                
                return True
                
            except Exception as e:
                log_error(f"連接錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}")
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                
                if attempt < max_retries - 1:
                    logg(f"[!] {retry_delay}秒後重試...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log_error("達到最大重試次數")
                    return False
        
        return False

    def run(self):
        """運行礦機"""
        while True:  # 添加外層循環
            try:
                if not self.connect():
                    logg("[!] 連接失敗，5秒後重試...")
                    time.sleep(5)
                    continue
                
                # 清空螢幕並初始化顯示
                print('\033[2J\033[H', end='')
                print('\033[?25l', end='')
                
                # 啟動挖礦線程
                self.mining_thread = threading.Thread(target=self.mine)
                self.mining_thread.daemon = True
                self.mining_thread.start()
                
                # 啟動網絡線程
                self.network_thread = threading.Thread(target=self.network_loop)
                self.network_thread.daemon = True
                self.network_thread.start()
                
                # 主線程處理狀態顯示
                try:
                    while self.working:
                        self.print_mining_stats()
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.working = False
                    print("\n[!] 正在停止挖礦...")
                    break  # 跳出外層循環
                
            except Exception as e:
                log_error(f"運行時錯誤: {str(e)}")
                if self.working:
                    logg("[!] 5秒後重試...")
                    time.sleep(5)
                else:
                    break
            finally:
                self.working = False
                print('\033[?25h', end='')

    def network_loop(self):
        """網絡通信循環"""
        reconnect_delay = 5
        last_keepalive = time.time()
        
        while self.working:
            try:
                message = self.recv_message()
                if not message:
                    continue
                
                if message.get('method') == 'mining.notify':
                    self.process_job(message)
                elif message.get('method') == 'mining.set_difficulty':
                    self.current_difficulty = message['params'][0]
                elif message.get('error'):
                    log_error(f"礦池錯誤: {message['error']}")
                
                # 發送保活消息
                current_time = time.time()
                if current_time - last_keepalive >= 30:
                    self.send_message({
                        'id': 10,
                        'method': 'mining.subscribe',
                        'params': []
                    })
                    last_keepalive = current_time
                
            except socket.timeout:
                continue
            except ConnectionError:
                if self.working:
                    log_error(f"[!] 連接斷開，{reconnect_delay}秒後重試...")
                    time.sleep(reconnect_delay)
                    if not self.connect():
                        reconnect_delay = min(reconnect_delay * 2, 60)
                    else:
                        reconnect_delay = 5
            except Exception as e:
                if self.working:
                    log_error(f"網絡錯誤: {str(e)}")
                    time.sleep(1)

    def stats_loop(self):
        """狀態顯示循環"""
        while self.working:
            try:
                self.print_mining_stats()
                time.sleep(0.5)
            except Exception as e:
                log_error(f"Stats error: {str(e)}")

    def process_job(self, job):
        """處理新的挖礦任務"""
        if not job or 'params' not in job:
            return
        
        try:
            params = job['params']
            if len(params) < 8:
                log_error(f"無效的工作參數: {params}")
                return
            
            # 停止當前挖礦線程
            self.working = False
            if self.current_mining_thread and self.current_mining_thread.is_alive():
                self.current_mining_thread.join(timeout=1)
            
            # 解析任務參數
            job_id = params[0]
            prevhash = bytes.fromhex(params[1])
            coinb1 = bytes.fromhex(params[2])
            coinb2 = bytes.fromhex(params[3])
            merkle_branches = [bytes.fromhex(h) for h in params[4]]
            version = struct.pack('<I', int(params[5], 16))
            nbits = bytes.fromhex(params[6])
            ntime = bytes.fromhex(params[7])
            
            # 解析區塊高度
            try:
                height_marker = b'\x03'
                marker_pos = coinb1.find(height_marker)
                if marker_pos != -1:
                    height_bytes = coinb1[marker_pos+1:marker_pos+4]
                    self.current_height = int.from_bytes(height_bytes, 'little')
                    logg(f"[*] 新區塊高度: {self.current_height}")
                else:
                    height_marker = b'\x04'
                    marker_pos = coinb1.find(height_marker)
                    if marker_pos != -1:
                        height_bytes = coinb1[marker_pos+1:marker_pos+5]
                        self.current_height = int.from_bytes(height_bytes, 'little')
                        logg(f"[*] 新區塊高度: {self.current_height}")
            except:
                pass
            
            # 生成 extranonce2
            extranonce2 = os.urandom(self.extranonce2_size)
            
            # 計算 merkle root
            coinbase = coinb1 + bytes.fromhex(self.extranonce1) + extranonce2 + coinb2
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle_root = coinbase_hash
            for branch in merkle_branches:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch).digest()).digest()
            
            self.job = {
                'id': job_id,
                'version': version,
                'prevhash': prevhash,
                'merkle_root': merkle_root,
                'ntime': ntime,
                'nbits': nbits,
                'extranonce2': extranonce2.hex()
            }
            
            logg(f"[*] 新工作 - 區塊:{self.current_height} | 難度:{format_difficulty_value(self.network_difficulty)}")
            
            # 重置狀態並啟動新的挖礦線程
            self.working = True
            self.submitted_shares.clear()
            
            self.current_mining_thread = threading.Thread(target=self.mine)
            self.current_mining_thread.daemon = True
            self.current_mining_thread.start()
            
        except Exception as e:
            log_error(f"處理任務失敗: {str(e)}")
            traceback.print_exc()

    def mine(self):
        """優化的挖礦主循環"""
        if not self.job:
            return
        
        try:
            header_bin = b''.join([
                self.job['version'],
                self.job['prevhash'],
                self.job['merkle_root'],
                self.job['ntime'],
                self.job['nbits']
            ])
            
            nonce = int.from_bytes(os.urandom(4), 'little')
            total_hashes = 0
            last_hash_update = time.time()
            start_time = time.time()
            
            cpu_cores = max(1, cpu_count() - 1)
            logg(f"[*] 開始挖礦 [{cpu_cores}核心]")
            
            with Pool(cpu_cores) as pool:
                while self.working:
                    try:
                        if not self.job:  # 檢查是否有工作
                            time.sleep(0.1)
                            continue
                        
                        batch_results = []
                        batch_size = 100
                        
                        for _ in range(cpu_cores):
                            batch_results.append(pool.apply_async(
                                check_hash_batch_optimized,
                                args=(
                                    header_bin,
                                    (nonce + _ * batch_size) & 0xFFFFFFFF,
                                    batch_size,
                                    self.current_difficulty
                                )
                            ))
                        
                        for result in batch_results:
                            try:
                                found, hashes = result.get(timeout=1)
                                total_hashes += hashes
                                self.optimizer.total_hashes = total_hashes
                                
                                if found:
                                    logg(f"[+] 找到Share! [{found[0][0]:08x}]")
                                    self.submit_share(found[0][0])
                            except Exception as e:
                                continue
                        
                        # 更新哈希率
                        current_time = time.time()
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            hashrate = total_hashes / elapsed
                            self.optimizer.hashrate = hashrate
                            if current_time - last_hash_update >= 2.0:
                                last_hash_update = current_time
                        
                        nonce = (nonce + cpu_cores * batch_size) & 0xFFFFFFFF
                        
                    except Exception as e:
                        log_error(f"挖礦錯誤: {str(e)}")
                        time.sleep(0.1)
                    
        except Exception as e:
            log_error(f"嚴重錯誤: {str(e)}")

    def print_mining_stats(self):
        """打印挖礦統計信息"""
        try:
            runtime = time.time() - self.optimizer.start_time
            shares_per_min = (self.optimizer.shares_found * 60) / runtime if runtime > 0 else 0
            
            # 清除當前行並打印新狀態
            print('\033[K', end='')  # 清除當前行
            status = (
                f"\r[DOGE] "
                f"B:{self.current_height} | "
                f"H:{format_hashrate(self.optimizer.hashrate)}H/s | "
                f"S:{self.optimizer.shares_found}({shares_per_min:.1f}/m) | "
                f"D:{format_difficulty_value(self.current_difficulty)}"
            )
            print(status, end='', flush=True)
        except Exception as e:
            log_error(f"狀態更新錯誤: {str(e)}")

    def send_message(self, msg):
        """發送消息到礦池"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not self.sock:
                    raise ConnectionError("未連接到礦池")
                
                data = json.dumps(msg) + '\n'
                self.sock.sendall(data.encode())
                return True
                
            except Exception as e:
                log_error(f"發送錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    try:
                        if not self.connect():
                            log_error("重新連接失敗")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    except:
                        pass
                else:
                    raise ConnectionError(f"Send error: {str(e)}")
        
        return False

    def recv_message(self):
        """從礦池接收消息"""
        if not self.sock:
            return None
        
        try:
            data = b''
            while self.working:
                try:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        self.sock = None  # 連接已關閉
                        raise ConnectionError("伺服器關閉連接")
                    data += chunk
                    
                    while b'\n' in data:
                        pos = data.find(b'\n')
                        line = data[:pos].decode('utf-8', errors='ignore')
                        data = data[pos + 1:]
                        
                        if line.strip():  # 忽略空行
                            try:
                                msg = json.loads(line)
                                logg(f"[DEBUG] 收到消息: {msg}")
                                return msg
                            except json.JSONDecodeError as e:
                                log_error(f"無效的 JSON ({str(e)}): {line}")
                                continue
                except socket.timeout:
                    return None
                except ConnectionError:
                    self.sock = None
                    raise
                except Exception as e:
                    log_error(f"接收錯誤: {str(e)}")
                    self.sock = None
                    raise ConnectionError(f"接收錯誤: {str(e)}")
        
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            log_error(f"接收錯誤: {str(e)}")
            self.sock = None
        return None

    def submit_share(self, nonce):
        """提交 share 到礦池"""
        if not self.job:
            return False
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not self.sock:
                    logg("[!] 連接已斷開，嘗試重新連接...")
                    if not self.connect():
                        raise ConnectionError("無法重新連接到礦池")
                
                submit = {
                    'id': 4,
                    'method': 'mining.submit',
                    'params': [
                        self.username,
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
                
                log_error(f"Share 提交超時 (嘗試 {attempt + 1}/{max_retries})")
                
            except ConnectionError as e:
                log_error(f"連接錯誤: {str(e)}")
                if attempt < max_retries - 1:
                    logg(f"[!] {retry_delay}秒後重試...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log_error("達到最大重試次數")
                    return False
                
            except Exception as e:
                log_error(f"提交 share 時出錯: {str(e)}")
                if attempt < max_retries - 1:
                    logg(f"[!] {retry_delay}秒後重試...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log_error("達到最大重試次數")
                    return False
        
        return False

def check_hash_batch_optimized(header_bin, start_nonce, count, difficulty):
    """優化的 Scrypt 挖礦算法"""
    results = []
    local_hashes = 0
    
    try:
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        target = int(max_target * difficulty)
        
        for nonce in range(start_nonce, start_nonce + count):
            header_with_nonce = header_bin + struct.pack('<I', nonce)
            
            hash_bin = hashlib.scrypt(
                password=header_with_nonce,
                salt=b'',
                n=1024,
                r=1,
                p=1,
                maxmem=2**30,
                dklen=32
            )
            
            hash_int = int.from_bytes(hash_bin, 'little')
            local_hashes += 1
            
            if hash_int <= target:
                results.append((nonce, hash_int, True))
                return results, local_hashes
            
            if local_hashes >= 100:  # 每批次處理100個哈希
                break
                
    except Exception as e:
        log_error(f"Hash error: {str(e)}")
    
    return results, local_hashes

def format_difficulty_value(diff):
    """格式化難度值顯示"""
    if diff is None or diff == 0:
        return "0.00"
        
    if diff >= 1e12:
        return f"{diff/1e12:.3f}T"
    elif diff >= 1e9:
        return f"{diff/1e9:.3f}G"
    elif diff >= 1e6:
        return f"{diff/1e6:.3f}M"
    elif diff >= 1e3:
        return f"{diff/1e3:.3f}K"
    elif diff >= 1:
        return f"{diff:.3f}"
    else:
        # 處理小於1的難度值
        if diff < 0.001:
            return f"{diff:.6f}"  # 顯示更多小數位
        else:
            return f"{diff:.3f}"

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

def generate_worker_name(address):
    """生成帶有隨機編號的礦工名稱"""
    worker_id = random.randint(1, 99)
    return f"{address}.{worker_id:02d}"  # 使用02d確保是兩位數

if __name__ == '__main__':
    freeze_support()
    
    # Dogecoin 地址
    address = "D5VzryiU3bY85o8ridNZqPjtx4KiEkibiy"
    worker_name = generate_worker_name(address)
    
    logg(f"Starting DOGE Solo Miner")
    logg(f"Mining Address: {address}")
    logg(f"Worker Name: {worker_name}")
    logg(f"Pool: doge.luckymonster.pro:5112")
    logg(f"Mode: SOLO")
    logg(f"Network Difficulty: 57.322 M")
    logg(f"Block Reward: 10000 DOGE")
    
    miner = DogeSoloMiner(worker_name)  # 使用帶有編號的礦工名稱
    try:
        miner.run()
    except KeyboardInterrupt:
        miner.working = False
        logg("Shutting down...")
    except Exception as e:
        log_error(f"Fatal error: {str(e)}")
        traceback.print_exc()