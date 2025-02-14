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

# 日誌設置
def logg(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_error(error_msg):
    logg(f"ERROR: {error_msg}")

class BTCSMiningOptimizer:
    def __init__(self):
        self.initialized = False
        self.total_shares = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.total_hashes = 0
        self.current_hashrate = 0
        self.shares_found = 0
        self.hashes_this_period = 0
        self.mining_active = False
        self.stats = {'hashrate': 0}
        logg("[*] BTCS Mining optimizer initialized")
        self.initialized = True
    
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

class BTCSSoloMiner:
    def __init__(self, address):
        # 修改連接設定為代理地址和端口
        self.server = ('mine.pool.sexy', 3333)  # 使用代理地址
        self.telegram_id = "5979533210"  # 您的 Telegram ID
        self.worker_id = str(random.randint(1, 99)).zfill(2)
        self.username = f"{self.telegram_id}.worker{self.worker_id}"  # 使用 TelegramID.worker 格式
        self.address = address
        self.sock = None
        self.working = True
        self.job = None
        self.optimizer = BTCSMiningOptimizer()
        self.current_height = 0
        self.submitted_shares = set()
        self.last_status_update = time.time()
        self.current_mining_thread = None
        self.mining_lock = threading.Lock()
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.network_difficulty = 60.90e6  # 60.90 M 網絡難度
        logg(f"[*] 礦工名稱: {self.username}")
    
    def connect(self):
        """連接到礦池"""
        try:
            if self.sock:
                self.sock.close()
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect(self.server)
            
            # 訂閱
            subscribe = {
                'id': 1,
                'method': 'mining.subscribe',
                'params': [
                    'btcminer/2.0.0',
                    None
                ]
            }
            self.send_message(subscribe)
            
            # 等待訂閱響應
            response = self.recv_message()
            if not response or response.get('error'):
                raise ConnectionError(f"Subscribe failed: {response.get('error') if response else 'No response'}")
            
            # 解析訂閱響應
            if response.get('result'):
                subscription = response['result']
                if isinstance(subscription, list) and len(subscription) >= 2:
                    self.extranonce1 = subscription[1]
                    self.extranonce2_size = subscription[2] if len(subscription) > 2 else 4
                    logg(f"[*] 訂閱成功 - ExtraNonce1: {self.extranonce1}")
                else:
                    raise ConnectionError("Invalid subscription response format")
            
            # 等待並處理難度設置
            message = self.recv_message()
            if message and message.get('method') == 'mining.set_difficulty':
                self.current_difficulty = message['params'][0]
                logg(f"[*] 設置難度: {self.current_difficulty}")
            
            # 授權
            authorize = {
                'id': 2,
                'method': 'mining.authorize',
                'params': [
                    self.username,
                    "x"
                ]
            }
            self.send_message(authorize)
            
            # 等待授權響應和歡迎消息
            got_auth = False
            got_welcome = False
            timeout = time.time() + 10
            
            while time.time() < timeout and not (got_auth and got_welcome):
                message = self.recv_message()
                if not message:
                    continue
                
                if message.get('id') == 2:
                    if message.get('result') is True:
                        got_auth = True
                        logg("[*] 授權成功")
                    elif message.get('error'):
                        raise ConnectionError(f"授權失敗: {message['error']}")
                elif message.get('method') == 'client.show_message':
                    msg = message['params'][0]
                    logg(f"[*] 礦池消息: {msg}")
                    if 'Authorised' in msg or 'authorized' in msg.lower():
                        got_welcome = True
                elif message.get('method') == 'mining.notify':
                    self.process_job(message)
                    logg("[*] 收到工作通知")
                
                time.sleep(0.1)
            
            if not got_auth or not got_welcome:
                raise ConnectionError("授權流程未完成")
            
            return True
            
        except Exception as e:
            log_error(f"Connection error: {str(e)}")
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            self.sock = None
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
        """發送消息到礦池"""
        try:
            data = json.dumps(msg) + '\n'
            self.sock.sendall(data.encode())
        except Exception as e:
            raise ConnectionError(f"Send error: {str(e)}")
    
    def recv_message(self, ignore_timeout=False):
        """從礦池接收消息"""
        try:
            data = b''
            while self.working:
                try:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("Connection closed by server")
                    data += chunk
                    
                    while b'\n' in data:
                        pos = data.find(b'\n')
                        line = data[:pos].decode('utf-8', errors='ignore')
                        data = data[pos + 1:]
                        
                        if line.strip():  # 忽略空行
                            try:
                                msg = json.loads(line)
                                logg(f"[DEBUG] 收到消息: {msg}")  # 添加調試信息
                                return msg
                            except json.JSONDecodeError as e:
                                log_error(f"Invalid JSON ({str(e)}): {line}")
                                continue
                except socket.timeout:
                    if not ignore_timeout:
                        return None
                    continue
            
        except ConnectionError as e:
            raise e
        except Exception as e:
            log_error(f"Receive error: {str(e)}")
        return None
    
    def reconnect(self):
        """重新連接到礦池"""
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        
        self.sock = None
        time.sleep(5)
        
        while self.working:
            try:
                if self.connect():
                    logg("[*] 重新連接成功")
                    return
            except Exception as e:
                log_error(f"Reconnection failed: {str(e)}")
                time.sleep(5)

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
            
            # 計算區塊難度
            try:
                nbits_int = int.from_bytes(nbits, 'little')
                exp = nbits_int >> 24
                mant = nbits_int & 0xffffff
                target_val = mant * (1 << (8 * (exp - 3)))
                self.network_difficulty = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 / target_val
                
                # 設置當前難度為礦池設置的難度 (0.001)
                self.current_difficulty = 0.001
                logg(f"[*] 區塊難度: {format_difficulty_value(self.network_difficulty)}")
                logg(f"[*] Share 難度: {self.current_difficulty}")
            except Exception as e:
                log_error(f"計算難度時出錯: {str(e)}")
                traceback.print_exc()
            
            # 生成 extranonce2
            extranonce2 = struct.pack('<I', int.from_bytes(os.urandom(4), 'little'))
            
            # 計算 merkle root
            coinbase = coinb1 + bytes.fromhex(self.extranonce1) + extranonce2 + coinb2
            coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
            
            merkle_root = coinbase_hash
            for branch in merkle_branches:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + branch).digest()).digest()
            
            # 更新任務信息
            self.job = {
                'id': job_id,
                'version': version,
                'prevhash': prevhash,
                'merkle_root': merkle_root,
                'ntime': ntime,
                'nbits': nbits,
                'extranonce2': extranonce2.hex(),
                'clean_jobs': params[8] if len(params) > 8 else False
            }
            
            logg(f"[*] 新工作 - 區塊:{self.current_height} | 難度:{format_difficulty_value(self.network_difficulty)} | Job ID:{job_id}")
            
            # 重置狀態並啟動新的挖礦線程
            self.working = True
            self.submitted_shares.clear()
            
            self.current_mining_thread = threading.Thread(target=self.mine)
            self.current_mining_thread.daemon = True
            self.current_mining_thread.start()
            logg("[*] 啟動新的挖礦線程")
            
        except Exception as e:
            log_error(f"處理任務失敗: {str(e)}")
            traceback.print_exc()

    def mine(self):
        """挖礦主循環"""
        if not self.job:
            logg("[*] 等待挖礦任務...")
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
            
            logg("[*] 開始挖礦計算...")
            
            while self.working:
                try:
                    result, hashes = check_hash_batch_optimized((
                        header_bin,
                        nonce,
                        1000,
                        0
                    ))
                    
                    total_hashes += hashes
                    self.optimizer.total_hashes = total_hashes
                    
                    # 更新哈希率
                    current_time = time.time()
                    if current_time - last_hash_update >= 0.1:
                        self.optimizer.update_hashrate(total_hashes)
                        last_hash_update = current_time
                    
                    if result:
                        logg(f"[*] 找到 share! Nonce: {result[0][0]:08x}")
                        self.submit_share(result[0][0])
                    
                    nonce = (nonce + 1000) & 0xFFFFFFFF
                    
                except Exception as e:
                    log_error(f"Mining iteration error: {str(e)}")
                    time.sleep(0.1)
                
        except Exception as e:
            log_error(f"Mining error: {str(e)}")
            traceback.print_exc()

    def run(self):
        """運行礦機"""
        while self.working:
            try:
                if not self.connect():
                    log_error("Connection failed, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # 清空螢幕並移動到頂部
                print('\033[2J\033[H', end='')
                print('\033[?25l', end='')  # 隱藏光標
                print('\033[s', end='')  # 保存光標位置
                self.print_mining_stats()
                print('\n', end='')
                
                last_status_update = 0
                last_keepalive = time.time()
                
                while self.working:
                    try:
                        # 處理消息
                        message = self.recv_message()
                        if message:
                            if message.get('method') == 'mining.notify':
                                self.process_job(message)
                            elif message.get('method') == 'mining.set_difficulty':
                                self.current_difficulty = message['params'][0]
                                logg(f"[*] 新的 Share 難度: {self.current_difficulty}")
                        
                        # 發送保活消息
                        current_time = time.time()
                        if current_time - last_keepalive >= 30:
                            self.send_message({'id': 10, 'method': 'mining.ping', 'params': []})
                            last_keepalive = current_time
                        
                        # 更新狀態顯示
                        if current_time - last_status_update >= 0.5:
                            print('\033[u', end='')
                            self.print_mining_stats()
                            print('\033[u\033[2B', end='')
                            last_status_update = current_time
                        
                        time.sleep(0.1)
                        
                    except socket.timeout:
                        continue
                    except ConnectionError as e:
                        log_error(f"Connection error: {str(e)}")
                        break
                    except Exception as e:
                        log_error(f"Error: {str(e)}")
                        break
                
            except KeyboardInterrupt:
                self.working = False
                print('\033[?25h', end='')
                break
            except Exception as e:
                log_error(f"Fatal error: {str(e)}")
                time.sleep(5)
    
    def print_mining_stats(self):
        """打印挖礦統計信息"""
        runtime = time.time() - self.optimizer.start_time
        shares_per_min = (self.optimizer.shares_found * 60) / runtime if runtime > 0 else 0
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 清除當前行並打印狀態
        print(f"\r\033[K{current_time} - [BTCS] "
              f"Block:{self.current_height} | "
              f"Diff:{format_difficulty_value(self.network_difficulty)} | "
              f"Speed:{format_hashrate(self.optimizer.hashrate)}H/s | "
              f"Shares:{self.optimizer.shares_found}({shares_per_min:.1f}/m) | "
              f"Total Hash:{format_hashes(self.optimizer.total_hash_count)}", 
              end='', flush=True)

    def submit_share(self, nonce):
        """提交 share 到礦池"""
        if not self.job:
            return
        
        try:
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
            
            log_error("Share 提交超時")
            return False
            
        except Exception as e:
            log_error(f"提交 share 時出錯: {str(e)}")
            traceback.print_exc()
            return False

    def verify_share(self, hash_int):
        """驗證 share 是否符合難度要求"""
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        share_target = int(max_target * self.current_difficulty)
        
        if hash_int <= share_target:
            logg(f"[DEBUG] Share 驗證通過")
            logg(f"[DEBUG] Hash: {hash_int:064x}")
            logg(f"[DEBUG] Target: {share_target:064x}")
            return True
        return False

def check_hash_batch_optimized(args):
    """優化的挖礦算法"""
    header_bin, start_nonce, count, target = args
    results = []
    local_hashes = 0
    
    try:
        # 使用固定的 share 難度 (0.001)
        difficulty = 0.001
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
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

def format_difficulty(diff):
    """格式化難度值，使用科學記數法"""
    return f"{diff:.2e}"

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
    freeze_support()
    
    address = "bc1q2gpqxejg95zredk2fz8f7kej2q9vkgkpe9sn8v"
    
    logg(f"Starting BTCS Miner")
    logg(f"Mining Address: {address}")
    logg(f"Pool: mine.pool.sexy:3333")  # 使用代理地址
    logg(f"Telegram ID: 5979533210")
    logg(f"Mode: PROXY")
    logg(f"Network Difficulty: 60.90 M")
    
    miner = BTCSSoloMiner(address)
    try:
        miner.run()
    except KeyboardInterrupt:
        miner.working = False
        logg("Shutting down...")
    except Exception as e:
        log_error(f"Fatal error: {str(e)}")
        traceback.print_exc() 