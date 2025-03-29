import os



def read_subfolders(path):
    # find all sub folders under target directory
    sub_folders = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    remove_list = [
        "remove",
    ]
    sub_folders = [ele for ele in sub_folders if ele not in remove_list]
    print(sub_folders)
    sub_folders = sorted(sub_folders, key=lambda s: float(s.split("_")[1]))

    return sub_folders