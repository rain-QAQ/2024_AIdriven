import sys
import os


class Logger(object):
    def __init__(self, filename="default.log", overwrite=False):
        self.terminal = sys.stdout
        filepath, _ = os.path.split(filename)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # change_by_zsy: 添加UTF-8编码支持，避免中文乱码
        self.log = open(filename, 'w' if overwrite else 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()
