import os
from tqdm import tqdm

########################  3D  #############################
SSL_train = '3D_images.txt'
SSL_train_f = open(SSL_train, 'a+')


img_folder_path1 = '3D_subvolumes/'
for root, dirs, files in os.walk(img_folder_path1):
    for i_files in tqdm(sorted(files)):
        img_path = os.path.join(root, i_files)
        result = img_path + '\n'
        SSL_train_f.write(result)


SSL_train_f.close()


########################  2D  #############################
SSL_train = '2D_images.txt'
SSL_train_f = open(SSL_train, 'a+')

img_folder_path1 = '2D_images/'
for root, dirs, files in os.walk(img_folder_path1):
    for i_files in tqdm(sorted(files)):
        img_path = os.path.join(root, i_files)
        result = img_path + '\n'
        SSL_train_f.write(result)

SSL_train_f.close()
