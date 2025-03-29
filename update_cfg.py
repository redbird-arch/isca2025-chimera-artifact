# File name  :    update_cfg.py
# Author     :    xiaocuicui
# Time       :    2024/07/24 15:58:46
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '../'))


def update_cfg_file(file_path, new_router_config, new_link_config):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        for line in lines:
            if 'dsent_router_config =' in line:
                line = f'dsent_router_config = {new_router_config};\n'
            elif 'dsent_link_config =' in line:
                line = f'dsent_link_config = {new_link_config};\n'
            file.write(line)

def update_all_cfg_files(directory, new_router_config, new_link_config):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cfg'):
                file_path = os.path.join(root, file)
                update_cfg_file(file_path, new_router_config, new_link_config)
                print(f'Updated: {file_path}')


directory = file_path
print(file_path)
new_router_config = os.path.join(file_path, 'src/communication/backend/booksim2/src/dsent/configs/dsent_router.cfg')
new_link_config = os.path.join(file_path, 'src/communication/backend/booksim2/src/dsent/configs/dsent_link.cfg')

update_all_cfg_files(directory, new_router_config, new_link_config)
