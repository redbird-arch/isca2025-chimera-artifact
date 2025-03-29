# File name  :    config.py
# Author     :    xiaocuicui
# Time       :    2024/06/30 10:51:15
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


def modify_topology_cfg_file(cfg_path, topology, k, n):

    with open(cfg_path, 'r') as file:
        lines = file.readlines()


    with open(cfg_path, 'w') as file:
        for line in lines:
            if line.startswith('topology ='):
                line = f'topology = {topology};\n'
            elif line.startswith('k ='):
                line = f'k = {k};\n'
            elif line.startswith('n ='):
                line = f'n = {n};\n'
            elif line.startswith('routing_function'):
                if topology == "mesh":
                    line = f'routing_function = dor;\n'
                elif topology == 'torus':
                    line = f'routing_function = dim_order;\n'
            file.write(line)


def modify_keyvalue_cfg_file(read_path, write_path, keyvalue_list):

    with open(read_path, 'r') as file:
        lines = file.readlines()
        file.close()
        
    with open(write_path, 'w') as file:
        for line in lines:
            for key_value in keyvalue_list:
                key = key_value[0]
                value = key_value[1]
                if line.startswith((key + " =")):
                    line = f'{key} = {value};\n'
                else:
                    continue
            file.write(line)
        file.close()




if __name__ == '__main__':

    cfg_path = os.path.join(file_path, '../communication/backend/booksim2/runfiles/mesh_o_torus_py.cfg')
    topology = 'torus'
    k = 8
    n = 3

    modify_topology_cfg_file(cfg_path, topology, k, n)






