import os
import glob

def extract_xml_filenames_to_txt(folder_path, output_file='./out_extract/val_zebra.txt'):

    xml_files = glob.glob(os.path.join(folder_path, '*.xml'))
    

    filenames = [os.path.basename(file) for file in xml_files]
    filenames = [ file[:-4] for file in filenames ]
    txt_content = []

    with open(output_file, 'w') as f:
        for filename in filenames:
            if filename in txt_content:
                continue
            txt_content.append(filename)

            f.write(filename + '\n')

folder_path = '/home/starkmar/Desktop/code/cc/out_extract/val_extract_unknow0_zebra'  
extract_xml_filenames_to_txt(folder_path)
