import os

def process_txt_files(folder_path):
    result = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            max_line_percentage = 0.0
            max_line_content = ""
            total_count = 0
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        count = int(parts[1].strip())
                        total_count += count
                
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        item = parts[0].strip()
                        count = int(parts[1].strip())
                        percentage = count / total_count
                        if percentage > max_line_percentage:
                            max_line_percentage = percentage
                            max_line_content = line.strip()
            
            result.append((filename, max_line_content, max_line_percentage))
    
    return result

folder_path = './tongji_180_train_new'  # 文件夹路径
log_file = "./bili_180_train_new.txt"
result = process_txt_files(folder_path)
for item in result:
    print(f"{item[0]}: {item[1]} - {item[2]:.2%}")

with open(log_file, 'w') as f:
    for item in result:
        f.write(f"{item[0]}: {item[1]} - {item[2]:.2%}\n")
