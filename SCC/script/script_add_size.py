import os
import xml.etree.ElementTree as ET

def add_size_info(filefolder1, filefolder2):
    for filename in os.listdir(filefolder1):
        if filename.endswith('.xml'):
            file1_path = os.path.join(filefolder1, filename)
            file2_path = os.path.join(filefolder2, filename)
            
            if os.path.exists(file2_path):
                tree1 = ET.parse(file1_path)
                root1 = tree1.getroot()
                
                tree2 = ET.parse(file2_path)
                size_element = tree2.find('.//size')
                
                if size_element is not None:
                    root1.insert(1, size_element)
                    
                    tree1.write(file1_path)
                    print(f"Size information added to {file1_path}")
                else:
                    print(f"No size information found in {file2_path}")

filefolder1_path = '/media/D/datasets/out_extract/val_extract_unknow0_zebra'
filefolder2_path = '/media/D/datasets/VOC2007/Annotations_beifen'

add_size_info(filefolder1_path, filefolder2_path)