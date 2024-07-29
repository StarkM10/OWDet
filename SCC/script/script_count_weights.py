
import os
from collections import defaultdict


def read_txt(txt_folder):
    data_all = {}
    for txt_file in os.listdir(txt_folder):
        data = [  ]
        with open(os.path.join(txt_folder, txt_file), 'r') as f:
            for line in f:
                data = line.split()
                if len(data) >= 6:
                    txt_xmin = float(data[2])
                    txt_ymin = float(data[3])
                    txt_xmax = float(data[4])
                    txt_ymax = float(data[5])
                    data.append( (txt_ymax-txt_ymin)/(txt_xmax-txt_xmin )  )
            # data_all.update( {txt_file, data} )
            data_all[txt_file] = data
    return data_all
        
# 将统计结果写入输出txt文件中
def write_output(output_folder, data_all):
    for key, value in data_all.items():
        with open(os.path.join(output_folder, key), 'w') as f:
            for line in value:
                f.write(f'{line}\n')


if __name__ == '__main__':
    txt_folder = './datasets/tongji_file'
    output_folder = './out/bili'
    os.makedirs(output_folder, exist_ok=True)
    data_all = read_txt (txt_folder)   
    write_output(output_folder, data_all)