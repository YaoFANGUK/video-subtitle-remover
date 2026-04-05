# coding: utf-8
import re
import os
import sys
import requests

from PySide6.QtCore import QVersionNumber

from backend.config import VERSION, PROJECT_UPDATE_URLS, tr


class VersionService:
    """ Version service """

    def __init__(self):
        self.current_version = VERSION
        self.lastest_version = VERSION
        self.version_pattern = re.compile(r'v*((\d+)\.(\d+)\.(\d+))')
        self.api_endpoints = PROJECT_UPDATE_URLS

    def get_latest_version(self):
        """ get latest version """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.64"
        }

        proxy = self.get_system_proxy()
        proxies = {
            "http": proxy,
            "https": proxy
        }

        # 依次尝试不同的API端点
        for url in self.api_endpoints:
            try:
                response = requests.get(url, headers=headers, timeout=5, allow_redirects=True, proxies=proxies)
                response.raise_for_status()
                
                # 解析版本
                version = response.json()['tag_name']  # type:str
                match = self.version_pattern.search(version)
                if not match:
                    continue  # 如果版本格式不匹配，尝试下一个API

                self.lastest_version = match.group(1)
                print(tr['VersionService']['VersionInfo'].format(VERSION, self.lastest_version))
                return self.lastest_version
            except Exception as e:
                print(tr['VersionService']['RequestError'].format(url, str(e)))
                continue  # 出错时尝试下一个API
        
        # 所有API都失败时返回当前版本
        return VERSION

    def has_new_version(self) -> bool:
        """ check whether there is a new version """
        version = QVersionNumber.fromString(self.get_latest_version())
        current_version = QVersionNumber.fromString(self.current_version)
        return version > current_version

    def get_system_proxy(self):
        """ get system proxy """
        if sys.platform == "win32":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
                    enabled, _ = winreg.QueryValueEx(key, 'ProxyEnable')

                    if enabled:
                        return "http://" + winreg.QueryValueEx(key, 'ProxyServer')
            except:
                pass
        elif sys.platform == "darwin":
            s = os.popen('scutil --proxy').read()
            info = dict(re.findall(r'(?m)^\s+([A-Z]\w+)\s+:\s+(\S+)', s))

            if info.get('HTTPEnable') == '1':
                return f"http://{info['HTTPProxy']}:{info['HTTPPort']}"
            elif info.get('ProxyAutoConfigEnable') == '1':
                return info['ProxyAutoConfigURLString']

        return os.environ.get("http_proxy")