import os
import xml.etree.ElementTree as ET

# 文件路径定义
txt_file_path = '/media/D/datasets/VOC2007/ImageSets/Main/all_task_val.txt'
folder1_path = '/media/D/datasets/VOC2007/Annotations'
folder2_path = '/media/D/datasets/VOC2007/Annotations_beifen'




with open(txt_file_path, 'r') as file:
    xml_files = file.read().splitlines()

for xml_filename in xml_files:
    xml_path_folder1 = os.path.join(folder1_path, xml_filename+".xml")
    xml_path_folder2 = os.path.join(folder2_path, xml_filename+".xml")
    
    if os.path.exists(xml_path_folder1):
        tree1 = ET.parse(xml_path_folder1)
        root1 = tree1.getroot()
        
        has_unknow0 = any(obj.find('name').text == 'unknow0' for obj in root1.findall('object'))
        if has_unknow0:
            if os.path.exists(xml_path_folder2):
                tree2 = ET.parse(xml_path_folder2)
                root2 = tree2.getroot()
                
                non_zebra_objs = [obj for obj in root2.findall('object') if obj.find('name').text != 'zebra']
                for obj in non_zebra_objs:
                    root1.append(obj)
                
                tree1.write(xml_path_folder1)
                print(f"Updated {xml_filename} with additional objects.")
            else:
                print(f"{xml_filename} not found in {folder2_path}.")
        else:
            print(f"No 'unknow0' object found in {xml_filename}.")
    else:
        print(f"{xml_filename} not found in {folder1_path}.")
