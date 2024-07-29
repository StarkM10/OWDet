
import os
import cv2
import xml.etree.ElementTree as ET

output_folder = "./new_images/unknow0/"
image_folder = "/media/D/datasets/VOC2007/JPEGImages/"
unknow_flag = "unknow0"
os.makedirs(output_folder, exist_ok=True)

image_boxes = {}

for xml_file in os.listdir("./out/cluster_label/"):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join("./out/cluster_label/", xml_file))
    root = tree.getroot()

    img_file = os.path.splitext(xml_file)[0]

    if img_file not in image_boxes:
        image_boxes[img_file] = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name == unknow_flag:
            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)
            image_boxes[img_file].append((xmin, ymin, xmax, ymax))

for img_file, boxes in image_boxes.items():
    img_path = os.path.join(image_folder, img_file + ".jpg")
    image = cv2.imread(img_path)

    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(image, unknow_flag, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = os.path.join(output_folder, img_file + ".jpg")
    cv2.imwrite(output_path, image)

    print(f"Processed {img_file}.jpg")

print("All images processed and saved.")
