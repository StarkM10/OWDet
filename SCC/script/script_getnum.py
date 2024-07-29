
import os
from collections import defaultdict

class_count = defaultdict(lambda: defaultdict(int))


def parse_xml(xml_file):
    with open(xml_file, 'r') as f:
        data = f.read()
    blocks = data.split('<object>')[1:]
    for block in blocks:
        xmin = float(block.split('<xmin>')[1].split('</xmin>')[0])
        ymin = float(block.split('<ymin>')[1].split('</ymin>')[0])
        xmax = float(block.split('<xmax>')[1].split('</xmax>')[0])
        ymax = float(block.split('<ymax>')[1].split('</ymax>')[0])
        class_name = block.split('<name>')[1].split('</name>')[0]
        yield xmin, ymin, xmax, ymax, class_name


def parse_txt(txt_folder, xmin, ymin, xmax, ymax):
    for txt_file in os.listdir(txt_folder):
        with open(os.path.join(txt_folder, txt_file), 'r') as f:
            for line in f:
                data = line.split()
                if len(data) >= 6:
                    txt_xmin = float(data[2])
                    txt_ymin = float(data[3])
                    txt_xmax = float(data[4])
                    txt_ymax = float(data[5])
                    if xmin == txt_xmin and ymin == txt_ymin and xmax == txt_xmax and ymax == txt_ymax:
                        return txt_file


def count_classes(xml_folder, txt_folder):
    for xml_file in os.listdir(xml_folder):
        print("正在处理"+xml_file+"文件")
        for xmin, ymin, xmax, ymax, class_name in parse_xml(os.path.join(xml_folder, xml_file)):
            txt_file = parse_txt(txt_folder, xmin, ymin, xmax, ymax)
            if txt_file:
                class_count[class_name][txt_file[:-4]] += 1


def write_output(output_folder):
    for class_name, class_data in class_count.items():
        with open(os.path.join(output_folder, f'{class_name}.txt'), 'w') as f:
            print("正在写入"+class_name+"文件")
            for txt_file, count in class_data.items():
                f.write(f'{txt_file}: {count}\n')


if __name__ == '__main__':
    xml_folder = './out/cluster_label_170_train'
    txt_folder = './datasets/t1_train_all'
    # txt_folder = './datasets/t1_test_all'
    # txt_folder = './datasets/t1_val_all'
    output_folder = './tongji_170_train'
    os.makedirs(output_folder, exist_ok=True)
    count_classes(xml_folder, txt_folder)
    write_output(output_folder)