import os
import argparse
import torch
import torchvision
import numpy as np
from torch.utils.data import Subset, SubsetRandomSampler

from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
import copy
from sklearn import metrics

from utils.CustomDataset import CustomDataset
from xml.dom import minidom


def inference(loader, model, device, mask):
    model.eval()
    feature_vector = []
    labels_vector = []
    # 轮廓系数
    silo_x = []
    # 标签
    block_image_map = []
    x_min_info = []
    y_min_info = []
    x_max_info = []
    y_max_info = []

    if mask is not None:
        mask_index = np.array(np.where(mask == 1))
        mask_ = torch.tensor(mask_index).squeeze().to(device)
        for step, (x, y, img, x_center, y_center, W, H) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                c, h = model.unk_forward_cluster(x, mask_)
            c = c.detach()
            h = h.detach()
            # silo_x.extend(h.cpu().detach().numpy())
            silo_x.extend(h.cpu())
            feature_vector.extend(c.cpu().detach().numpy())
            labels_vector.extend(y.numpy())
            # 标签信息
            block_image_map.append(img)
            x_min_info.append(x_center)
            y_min_info.append(y_center)
            x_max_info.append(W)
            y_max_info.append(H)
            pass

            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")
    else:

        for step, (x, y, img, x_center, y_center, W, H) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                c, h = model.forward_cluster(x)
            c = c.detach()
            feature_vector.extend(c.cpu().detach().numpy())
            labels_vector.extend(y.numpy())
            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")
    # silo_x = np.array(silo_x)
    feature_vector = np.array(feature_vector)
    # mar
    # for ii in range(60):
    labels_vector = np.array(labels_vector)
    if mask is not None:
        labels_vector = labels_vector
    print("Features shape {}".format(feature_vector.shape))
    # print(block_image_map)
    print("block_image_map:"+str(len(block_image_map)))
    return feature_vector, labels_vector, silo_x, block_image_map, x_min_info, y_min_info, x_max_info, y_max_info


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/t1_new_5.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 200
    elif args.dataset == "sub_voc":
        dataset = torchvision.datasets.ImageFolder(
            root='/media/F/sub_voc/test',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        unk_dataset = torchvision.datasets.ImageFolder(
            root='/media/F/sub_voc/test_unk_60',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 80
    elif args.dataset == "mcc":

        dataset = CustomDataset(
            txt_dir='./datasets/t1_train_all',
            transform=transform.Transforms(size=args.image_size).test_transform,
            test=True,
        )
        unk_dataset = CustomDataset(
            txt_dir='./datasets/t1_train_unknown_class',
            transform=transform.Transforms(size=args.image_size).test_transform,
            test=False,
            unk_test=True
        )

        # dataset = CustomDataset(
        #     txt_dir='./datasets/t1_test',
        #     transform=transform.Transforms(size=args.image_size).test_transform,
        #     test=True,
        # )
        # unk_dataset = CustomDataset(
        #     txt_dir='./datasets/t1_test_unknow',
        #     transform=transform.Transforms(size=args.image_size).test_transform,
        #     test=False,
        #     unk_test=True
        # )

        # dataset = CustomDataset(
        #     txt_dir='./datasets/t1_val',
        #     transform=transform.Transforms(size=args.image_size).test_transform,
        #     test=True,
        # )
        # unk_dataset = CustomDataset(
        #     txt_dir='./datasets/t1_val_unknow',
        #     transform=transform.Transforms(size=args.image_size).test_transform,
        #     test=False,
        #     unk_test=True
        # )

        class_num = args.cats
    else:
        raise NotImplementedError

    # Compute the size of the subset to sample
    subset_size = len(dataset) // 5

    # Create a subset random sampler object
    indices = list(range(len(dataset)))
    subset_indices = indices[::5]  # select every 5th index
    sampler = SubsetRandomSampler(subset_indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        sampler=None,
        drop_last=False,
        num_workers=args.workers,
    )
    unk_data_loader = torch.utils.data.DataLoader(
        unk_dataset,
        batch_size=500,
        shuffle=False,
        sampler=None,
        drop_last=False,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    print(model_fp)
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y, silo_x, block_image_map0, x_min_info0, y_min_info0, x_max_info0, y_max_info0 = inference(data_loader, model,
                                                                                                   device, mask=None)
    mask = evaluation.get_mask(Y, X)
    X, Y, silo_xx, block_image_map, x_min_info, y_min_info, x_max_info, y_max_info = inference(unk_data_loader, model,
                                                                                               device, mask)
    #  X是各个样本的标签类别号   Y是各个样本的聚类预测结果类别号
    # 类别数
    print("Xnum: start" + str(len(X)))
    print("block_num: start" + str(len(block_image_map)))
    # print(block_image_map)
    num_classes = 60
    x1 = []
    for i in range(len(X)):
        silo_sub = silo_xx[i].tolist()
        x1.append(silo_sub)
    x2 = torch.tensor(X).tolist()
    silo_result = metrics.silhouette_samples(x1, x2, metric='euclidean')
    silo_result_cluster = [[] for _ in range(num_classes)]
    cls_imgnum_map = [[] for _ in range(num_classes)]
    for i in range(len(X)):
        label = X[i].item()
        silo_result_cluster[label].append(silo_result[i])
        cls_imgnum_map[label].append(i)
    silo_result_cluster_avg = []
    silo_cluster_number = []
    for class_idx in range(num_classes):
        cls_sum = sum(silo_result_cluster[class_idx])
        cls_num = len(silo_result_cluster[class_idx])
        silo_cluster_number.append(cls_num)
        if cls_num == 0:
            average = 0
        else:
            average = cls_sum / cls_num
        silo_result_cluster_avg.append(average)

    # 使用sorted函数对列表进行排序，reverse=True表示降序排序
    sorted_indices = sorted(range(len(silo_result_cluster_avg)), key=lambda x: silo_result_cluster_avg[x], reverse=True)

    # 获取排序前的前五个元素及其下标
    # top_five_values = [silo_result_cluster_avg[i] for i in sorted_indices[:5]]
    # top_five_indices = sorted_indices[:5]

    top_all_values = [silo_result_cluster_avg[i] for i in sorted_indices[:]]
    top_all_indices = sorted_indices[:]
    top_all_num = [silo_cluster_number[i] for i in sorted_indices[:]]

    # 初始化结果列表
    top_five_values = []
    top_five_indices = []
    top_five_num = []

    count = 0
    for i in range(len(top_all_values)):
        if top_all_num[i] >= 300 and count < 5:
            top_five_values.append(top_all_values[i])
            top_five_indices.append(top_all_indices[i])
            top_five_num.append(top_all_num[i])
            count += 1
        if count == 5:
            break

    print("最大的五个数:", top_five_values)
    print("最大的五个数的下标:", top_five_indices)
    print("block_num :" + str(len(block_image_map)))
    # block_image_map = [int(item) for item in block_image_map[i] for i in len(block_image_map)]
    result = []
    for item in block_image_map:
        result.extend(item)
    result = [int(x) for x in result]
    block_image_map = result
    block_image_map = [str(item).zfill(12) for item in block_image_map]
    # print(x_min_info)
    # print(str(len(x_min_info)))
    result_xmin = []
    result_xmax = []
    result_ymin = []
    result_ymax = []
    for item in x_min_info: 
        result_xmin.extend(item.tolist())   
    x_min_info = result_xmin
    for item in y_min_info: 
        result_ymin.extend(item.tolist())   
    y_min_info = result_ymin
    for item in x_max_info: 
        result_xmax.extend(item.tolist())   
    x_max_info = result_xmax
    for item in y_max_info: 
        result_ymax.extend(item.tolist())   
    y_max_info = result_ymax
    # x_min_info = x_min_info[0].tolist()
    # y_min_info = y_min_info[0].tolist()
    # x_max_info = x_max_info[0].tolist()
    # y_max_info = y_max_info[0].tolist()
    # print(x_min_info)
    # print(str(len(x_min_info)))

    # selected_indices = []
    # for i in range(num_classes):  # 遍历5个最优聚类
    #     cls_num = top_five_indices[i]  # 找到最优聚类编号
    #     cls_len = len(cls_imgnum_map[cls_num])  # 找到该属于该类的图像块的数量
    #     print(str(i)+": sample nums"+str(cls_len))
    #     if cls_len > 1:
    #         selected_indices.append(cls_num)



    save_path_dir = "./out/cluster_label_170_train"
    os.makedirs(save_path_dir, exist_ok=True)
    # X, Y, silo_xx, block_image_map, x_min_info, y_min_info, x_max_info, y_max_info
    out_filename = "./out/out_filename_val.txt"
    with open(out_filename, "w") as outfile:
        for i in range(5):  # 遍历5个最优聚类
            cls_num = top_five_indices[i]  # 找到最优聚类编号
            cls_len = len(cls_imgnum_map[cls_num])  # 找到该属于该类的图像块的数量
            
            # change id
            # if cls_num == 0:
            #     flag = 0
            # elif cls_num == 1:
            #     flag = 1
            # else:
            #     flag = i
            
            for j in range(cls_len):  # 分别遍历每个图像块
                img_num = cls_imgnum_map[cls_num][j]  # 找到该图像块在inference返回中的编号
                # print("当前为第" + str(i) + "个类,类编号为：" + str(cls_num) + ",len为：" + str(cls_len) + "，正在遍历其第" + str(j) + "个，块编号为" + str(img_num))
                # 以下分别为该图像块在原图像中的标签信息：
                img_gt = Y[img_num]
                img_name = block_image_map[img_num]
                img_x_min = x_min_info[img_num]
                img_y_min = y_min_info[img_num]
                img_x_max = x_max_info[img_num]
                img_y_max = y_max_info[img_num]
                # outfile.write(img_name + "\n")
                # 该图像块标签保存到xml中
                xml_path = save_path_dir + "/" + str(img_name) + ".xml"
                is_exist = os.path.exists(xml_path)
                if is_exist:  # 已经存在该文件：
                    with open(xml_path, "r", encoding="utf-8") as f:
                        original_content = f.read()

                    doc = minidom.parseString(original_content)
                    root_node = doc.documentElement
                    # 创建图像块节点 保存信息
                    object_node = doc.createElement("object")
                    root_node.appendChild(object_node)
                    # 添加未知类名 unknow i
                    object_name_node = doc.createElement("name")
                    object_name_value = doc.createTextNode(f'unknow{i}')
                    object_name_node.appendChild(object_name_value)
                    object_node.appendChild(object_name_node)
                    # 添加difficult
                    difficult_node = doc.createElement("difficult")
                    difficult_value = doc.createTextNode("0")
                    difficult_node.appendChild(difficult_value)
                    object_node.appendChild(difficult_node)
                    # 添加图像块边框
                    object_box_node = doc.createElement("bndbox")
                    object_node.appendChild(object_box_node)
                    # 添加xmin ymin xmax ymax值
                    x_min_node = doc.createElement("xmin")
                    x_min_value_node = doc.createTextNode(str(img_x_min))
                    y_min_node = doc.createElement("ymin")
                    y_min_value_node = doc.createTextNode(str(img_y_min))
                    x_max_node = doc.createElement("xmax")
                    x_max_value_node = doc.createTextNode(str(img_x_max))
                    y_max_node = doc.createElement("ymax")
                    y_max_value_node = doc.createTextNode(str(img_y_max))
                    x_min_node.appendChild(x_min_value_node)
                    y_min_node.appendChild(y_min_value_node)
                    x_max_node.appendChild(x_max_value_node)
                    y_max_node.appendChild(y_max_value_node)
                    object_box_node.appendChild(x_min_node)
                    object_box_node.appendChild(y_min_node)
                    object_box_node.appendChild(x_max_node)
                    object_box_node.appendChild(y_max_node)
                    with open(xml_path, "w", encoding="utf-8") as f:
                        f.write(doc.toxml())
                    
                else:
                    # 1. 创建对象
                    doc = minidom.Document()
                    # 2. 创建根结点，并用dom对象添加根结点
                    root_node = doc.createElement("annotation")
                    doc.appendChild(root_node)
                    # 3. 创建根结点
                    filename_node = doc.createElement("filename")
                    filename_value = doc.createTextNode(f'{img_name}.xml')
                    filename_node.appendChild(filename_value)
                    root_node.appendChild(filename_node)
                    # 4.创建图像块节点 保存信息
                    object_node = doc.createElement("object")
                    root_node.appendChild(object_node)
                    # 添加未知类名 unknow i
                    object_name_node = doc.createElement("name")
                    object_name_value = doc.createTextNode(f'unknow{i}')
                    object_name_node.appendChild(object_name_value)
                    object_node.appendChild(object_name_node)
                    # 添加difficult
                    difficult_node = doc.createElement("difficult")
                    difficult_value = doc.createTextNode("0")
                    difficult_node.appendChild(difficult_value)
                    object_node.appendChild(difficult_node)
                    # 添加图像块边框
                    object_box_node = doc.createElement("bndbox")
                    object_node.appendChild(object_box_node)
                    # 添加xmin ymin xmax ymax值
                    x_min_node = doc.createElement("xmin")
                    x_min_value_node = doc.createTextNode(str(img_x_min))
                    y_min_node = doc.createElement("ymin")
                    y_min_value_node = doc.createTextNode(str(img_y_min))
                    x_max_node = doc.createElement("xmax")
                    x_max_value_node = doc.createTextNode(str(img_x_max))
                    y_max_node = doc.createElement("ymax")
                    y_max_value_node = doc.createTextNode(str(img_y_max))
                    x_min_node.appendChild(x_min_value_node)
                    y_min_node.appendChild(y_min_value_node)
                    x_max_node.appendChild(x_max_value_node)
                    y_max_node.appendChild(y_max_value_node)
                    object_box_node.appendChild(x_min_node)
                    object_box_node.appendChild(y_min_node)
                    object_box_node.appendChild(x_max_node)
                    object_box_node.appendChild(y_max_node)
                    with open(xml_path, "w", encoding="utf-8") as f:
                        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i
    nmi, ari, f, acc, pur = evaluation.evaluate(Y, X)
    # 计算轮廓系数
    # silhouette_avg1 = metrics.silhouette_score(Y, X)
    # data1 = np.concatenate(data1, axis=0)
    # data1 = np.concatenate([t.cpu().numpy() for t in data1], axis=0)
    # silhouette_avg2 = metrics.silhouette_score(X.reshape(-1, 1), Y.reshape(-1, 1))
    # silhouette_avg2 = metrics.silhouette_score(data1.cpu().numpy(), X)
    # print('silhouette_avg1:',silhouette_avg1)
    # print('silhouette_avg2:', silhouette_avg2)
    # data参数表示数据样本的特征向量。它是一个二维数组或矩阵，其中每一行代表一个样本，每一列代表样本的一个特征。

    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f} PUR = {:.4f}'.format(nmi, ari, f, acc, pur))

    top_five_values_str = ', '.join(str(value) for value in top_five_values)
    top_five_indices_str = ', '.join(str(value) for value in top_five_indices)
    with open('./result.txt','a') as ff:
        ff.write("\n")
        ff.write(str(args.start_epoch)+":\n")
        ff.write("最大的五个数:"+top_five_values_str+"\n")
        ff.write("最大的五个数的下标:"+top_five_indices_str+"\n")
        ff.write('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f} PUR = {:.4f}'.format(nmi, ari, f, acc, pur) +"\n")

