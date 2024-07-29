import os
import xml.etree.ElementTree as ET

def read_txt_coords(filename):
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            coords.append((int(parts[0]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
    return coords

def filter_xml(xml_file, target_name, txt_coords, output_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects_to_keep = [obj for obj in root.findall('object') if obj.find('name').text == target_name]

    matched_objects = []
    
    for obj in objects_to_keep:
        xmin = float(obj.find('.//xmin').text)
        ymin = float(obj.find('.//ymin').text)
        xmax = float(obj.find('.//xmax').text)
        ymax = float(obj.find('.//ymax').text)
        if any((file_id, xmin, ymin, xmax, ymax) in txt_coords for file_id, _, _, _, _ in txt_coords):
            matched_objects.append(obj)

    if matched_objects:
        new_root = ET.Element('annotation')
        new_root.append(root.find('filename'))
        for obj in matched_objects:
            new_root.append(obj)
        
        new_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        tree = ET.ElementTree(new_root)
        tree.write(new_filename)
        print(f"Filtered XML saved as: {new_filename}")

def process_cluster_label(name, category_name, mode):
    output_folder = f"./out_extract/{mode}_extract_{name}_{category_name}"
    os.makedirs(output_folder, exist_ok=True)

    txt_file_path = os.path.join(f"./datasets/t1_{mode}_all/", f"{category_name}.txt")
    cluster_label_folder = f"./out/cluster_label_180_{mode}"
    txt_coords = read_txt_coords(txt_file_path)
    for xml_file in os.listdir(cluster_label_folder):
        if xml_file.endswith('.xml'):
            full_xml_path = os.path.join(cluster_label_folder, xml_file)
            filter_xml(full_xml_path, name, txt_coords, output_folder)

process_cluster_label('unknow4', 'toilet', 'val')