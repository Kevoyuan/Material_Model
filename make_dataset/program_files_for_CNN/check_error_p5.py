import os

parent_dir = './roughmodel_mesh4/keyword'

folder_names = [] # initialize empty list to store all folder names




for dirName, subdirList, fileList in os.walk(parent_dir):



    for fname in fileList:
        if "x-coordinate_0.csv" not in fileList:
            file_path = os.path.join(dirName, fname)
            

            print("Folders with no x-coordinate_0.csv file: ", dirName)
            