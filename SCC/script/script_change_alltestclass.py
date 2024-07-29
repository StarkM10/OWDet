import os

txt_file_path = '/media/D/datasets/VOC2007/ImageSets/Main/all_task_test.txt'
folder_path = '/media/D/datasets/VOC2007/Annotations'


with open(txt_file_path, 'r') as file:
    xml_files = file.read().splitlines()

for xml_file_name in xml_files:
    xml_file_path = os.path.join(folder_path, xml_file_name+".xml")
    
    if os.path.exists(xml_file_path):
        with open(xml_file_path, 'r') as xml_file:
            content = xml_file.read()
        
        new_content = content.replace('<name>zebra</name>', '<name>unknow0</name>')
        
        if content != new_content:
            with open(xml_file_path, 'w') as xml_file:
                xml_file.write(new_content)  
                print(f"Processed {xml_file_name}")
    else:
        print(f"File not found: {xml_file_name}")

print("All files processed.")
