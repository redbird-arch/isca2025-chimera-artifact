# File name  :    func.py
# Author     :    xiaocuicui
# Time       :    2024/07/05 14:15:58
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


division_list = lambda a, b: [b] * (a // b) + ([a % b] if a % b != 0 else [])




