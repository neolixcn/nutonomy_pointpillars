import numpy as np
import os


def txt_name_format(txt_path):
    """
    rename txt files: 0.txt --> 000000.txt
    """
    for txt in os.listdir(txt_path):
        print(txt_path + txt)
        print(txt_path + '%06d.bin'%int(txt.strip('.bin')))
        os.rename(txt_path + txt, txt_path + '%06d.bin'%int(txt.strip('.bin')))


def split_datasets(ids_p, num):
    ids = []
    for id_name in os.listdir(ids_path):
        ids.append(id_name.strip(".txt"))
    ids.sort()
    print(ids)
    train_num = len(ids) * 4 // 5
    train_set = set()
    train_val_set = set(ids)
    while len(train_set) < train_num:
        train_set.add(ids[np.random.randint(0, num)])
    val_set = train_val_set - train_set
    train_ls = list(train_set)
    val_ls = list(val_set)
    train_ls.sort()
    val_ls.sort()
    str_train_ids = ""
    for train_id in train_ls:
        str_train_ids += train_id + "\n"
    str_train_ids = str_train_ids.strip()
    with open('train_ids.txt', 'w') as f:
        f.write(str_train_ids)

    str_val_ids = ""
    for val_id in val_ls:
        str_val_ids += val_id + "\n"
    str_val_ids = str_val_ids.strip()
    with open('val_ids.txt', 'w') as f:
        f.write(str_val_ids)

    str_train_val_ids = ""
    for train_val_id in ids:
        str_train_val_ids += train_val_id + "\n"
    str_train_val_ids = str_train_val_ids.strip()
    with open("trainval_ids.txt", 'w') as f:
        f.write(str_train_val_ids)




if __name__ == '__main__':
    ids_path = "/nfs/nas/datasets/songhongli/neolix_shanghai_1924/training/label_2/"
    num = 1924
    split_datasets(ids_path, num)