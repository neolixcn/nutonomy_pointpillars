from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_recall_curve

def count_labels(l_path, cls):
    """
    size: h,w,l
    z_axis: the coordinate of the bounding box in z-axis
    """
    size = []
    z_axis = []
    for l in os.listdir(l_path):
        with open(l_path+l) as f:
            label_lines = f.readlines()
            for label_line in label_lines:
                label_line = label_line.split(" ")
                if label_line[0] == cls:
                    size.append([float(label_line[8]), float(label_line[9]), float(label_line[10])])
                    z_axis.append(float(label_line[13]))
                # if (float(label_line[8])>3)|(float(label_line[9])>3)|(float(label_line[10])>3):
                #     print("the vehicle is in %s" % l_path+l)
    np_size = np.array(size)
    np_z_axis = np.array(z_axis)
    return np_size.shape[0]
    # print("the number of %s is %d" % (cls, np_size.shape[0]))
    # print("the mean height of %s" % cls, np_size[:, 0].mean())
    # print("the mean width of %s" % cls, np_size[:, 1].mean())
    # print("the mean length of %s" % cls, np_size[:, 2].mean())
    # print("the mean z coordinate of %s" % cls, np_z_axis.mean())

def visu_class(class_path):
    plt.figure(figsize=(10, 10), dpi=80)
    N = 14
    class_ls = ["adult", "animal", "barrier", "bicycle", "bicycles", "bus", "car", "child", "cyclist", "dontcare",
                "motorcycle", "motorcyclist", "tricycle", "truck"]
    objects_num = []
    for c in class_ls:
        objects_num.append(count_labels(class_path, c))
    values = tuple(objects_num)
    index = np.arange(14)
    width = 0.45
    label_content = ""
    for c, n in zip(class_ls, objects_num):
        label_content += "%s: %d" % (c, n) + "\n"
    p2 = plt.bar(index, values, width, label=label_content, color="BLUE")
    plt.xlabel('class')
    plt.ylabel('number of bounding box')
    plt.title('000000-003025 Bounding box Distribution')
    plt.xticks(index, (class_ls))
    plt.legend(loc="upper right")
    plt.show()


def visu_PR_Curve():
    print("the first thresholds set")
    plt.figure("P-R Curve")
    plt.title("3D Precision/Recall Curve in first thresholds set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    class_num = 4
    for i in range(class_num):
        t_path = "/home/shl/data/PR_Curve/3026_aug/3D/%d.0.0_true.npy" % i
        s_path = "/home/shl/data/PR_Curve/3026_aug/3D/%d.0.0_score.npy" % i
        t_npy = np.load(t_path)
        s_npy = np.load(s_path)
        p, r, t = precision_recall_curve(t_npy, s_npy)
        if i == 0:
            l0, = plt.plot(r, p)
        elif i == 1:
            l1, = plt.plot(r, p)
        elif i == 2:
            l2, = plt.plot(r, p)
        elif i == 3:
            l3, = plt.plot(r, p)
    plt.legend(handles=[l0, l1, l2, l3], labels=['Pedestrian', 'Vehicle', 'Cyclist', 'Unknown'], loc='best')
    plt.show()


def visu_score_true(s_path, t_path):
    score = np.load(s_path)
    tr = np.load(t_path)
    pr_dict = {}
    for i in range(score.shape[0]):
        pr_dict[i] = score[i]
    new_pr_list = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)
    print([x[1] for x in new_pr_list])
    print([tr[x[0]] for x in new_pr_list])
    print(len(new_pr_list))
    # for i in range(score.shape[0]):
    #     print(score[i], tr[i])

if __name__ == "__main__":
    # visu_PR_Curve()
    # label_path = "/data/data/dataset/shanghai_to_annotate/shanghai_puruan/14cls_txts/"
    label_path = "/data/data/dataset/shanghai_to_annotate/shanghai_puruan/14cls_txts/"
    visu_class(label_path)
