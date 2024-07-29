# import os
#
# # 源文件夹路径
# source_folder = '/media/D/Lee/paper_code/OWDet/output_copy0/t1_task_unknown_class'
# # 目标文件夹路径
# target_folder = '/media/D/Lee/paper_code/OWDet/output_copy0/t1_task_unknown_class_top10'
#
# # 确保目标文件夹存在
# os.makedirs(target_folder, exist_ok=True)
#
# # 遍历源文件夹中的所有文件
# for filename in os.listdir(source_folder):
#     if filename.endswith('.txt'):
#         # 构建源文件路径和目标文件路径
#         source_file = os.path.join(source_folder, filename)
#         target_file = os.path.join(target_folder, filename)
#
#         # 打开源文件和目标文件
#         with open(source_file, 'r') as source, open(target_file, 'w') as target:
#             # 读取前10行内容
#             lines = source.readlines()[:10]
#
#             # 将前10行内容写入目标文件
#             target.writelines(lines)

# import os
#
# # 源文件夹路径
# source_folder = '/media/D/Lee/paper_code/OWDet/output_copy0/t1_task_unknown_class_top10'
# # 目标文件路径
# target_file = '/media/D/Lee/paper_code/OWDet/output_copy0/t1_task_unknown_class_top10_name.txt'
#
# # 遍历源文件夹中的所有文件
# for filename in os.listdir(source_folder):
#     if filename.endswith('.txt'):
#         # 构建源文件路径
#         source_file = os.path.join(source_folder, filename)
#
#         # 打开源文件
#         with open(source_file, 'r') as source:
#             # 读取前10行内容
#             lines = source.readlines()[:10]
#
#             # 提取每一行第一个空格之前的内容
#             extracted_lines = [line.split(' ')[0] for line in lines]
#
#             # 追加到目标文件中
#             with open(target_file, 'a') as target:
#                 for line in extracted_lines:
#                     target.write(line + '\n')

import os
from collections import defaultdict
import operator

# 文件路径
file_path = '/media/D/Lee/paper_code/OWDet/output_copy0/t1_task_unknown_class_top10_name.txt'

# 使用defaultdict创建一个默认值为0的字典
word_count = defaultdict(int)

# 打开文件
with open(file_path, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 去除行尾的换行符
        line = line.strip()

        # 以空格为分隔符将行拆分为单词
        words = line.split()

        # 统计每个单词的出现次数
        for word in words:
            word_count[word] += 1

# 按字典值（出现次数）进行降序排序
sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

# 打印结果
for word, count in sorted_word_count:
    print(f'{word}: {count}')