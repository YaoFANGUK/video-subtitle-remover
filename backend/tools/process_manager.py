# -*- coding: utf-8 -*-
"""
@desc: 进程管理器，用于管理和终止子进程
"""
import weakref
import signal
import os
import platform
import logging
import atexit
import subprocess
import concurrent.futures

class ProcessManager:
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = ProcessManager()
        return cls._instance
    
    def __init__(self):
        """初始化进程管理器"""
        self.processes = {}
        self.logger = logging.getLogger(__name__)
        
        # 注册退出处理函数
        atexit.register(self.terminate_all)
    
    def add_process(self, process, name=None):
        """
        添加进程到管理器
        
        Args:
            process: 要添加的进程对象 (subprocess.Popen实例)
            name: 进程名称，如果不提供则使用进程ID
        """
        if process is None:
            return
            
        process_id = name or f"Process:{id(process)}"
        self.processes[process_id] = process
        print(f"Added process: {process_id}, PID: {process.pid if hasattr(process, 'pid') else 'unknown'}")
        return process_id

    def add_pid(self, pid, name=None):
        process_id = name or f"Pid:{pid}"
        self.processes[process_id] = pid
        print(f"Added process: {process_id}, PID: {pid}")
        return process_id
    
    def remove_process(self, process_id):
        """
        从管理器中移除进程
        
        Args:
            process_id: 进程ID或名称
        """
        if process_id in self.processes:
            del self.processes[process_id]
            print(f"Removed process: {process_id}")
            return True
        return False
    
    def terminate_all(self):
        """并发终止所有管理的进程"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for process_id, process in list(self.processes.items()):
                if isinstance(process, int):
                    futures.append(executor.submit(self.terminate_by_pid, process))
                else:
                    futures.append(executor.submit(self.terminate_by_process, process))
            
            # 等待所有终止操作完成
            concurrent.futures.wait(futures)
        
        # 清空进程字典
        self.processes.clear()
    
    def terminate_by_process(self, process):
        if process is None:
            return
        try:
            print(f"Terminating process: pid: {process.pid}")
            if hasattr(process, 'poll') and process.poll() is not None:
                # 进程已经结束，直接返回
                return
                
            # 进程还在运行
            process.terminate()
            if hasattr(process, 'join'):
                try:
                    process.join(timeout=3)
                except:
                    pass
            if hasattr(process, 'wait'):
                try:
                    process.wait(timeout=3)
                except:
                    pass
            # 进程未能正常终止，尝试强制终止
            if hasattr(process, 'kill'):
                process.kill()
        except Exception as e:
            # print(f"Error terminating process: {str(e)}")
            pass
        self.terminate_by_pid(process.pid)

    def terminate_by_pid(self, pid):
        try:
            # 使用系统命令强制终止进程
            if platform.system() == 'Windows':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            else:
                subprocess.run(['pkill', '-9', '-P', str(pid)], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                subprocess.run(['kill', '-9', str(pid)], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        except Exception as e:
            print(f"Error forcibly terminating process with PID {pid}: {str(e)}")