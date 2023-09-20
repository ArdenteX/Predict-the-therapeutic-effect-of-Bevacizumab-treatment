import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


def split_image(path, base_path):
    print("Receive path: {} \n".format(path))
    # Get the OpenSlide object
    svs = openslide.OpenSlide(path)
    # Get the DeepZoomGenerator object to obtain the multiple level revolution radio
    dzg = DeepZoomGenerator(svs, tile_size=1000, limit_bounds=True, overlap=0)

    # 100um
    tiles = dzg.level_tiles[-2]

    num_h = tiles[0]
    num_w = tiles[1]

    # The higher the level of the DeepZoom, the greater multiple of the image
    for i in tqdm(range(num_h+1)):
        for j in range(num_w+1):
            try:
                img = np.array(dzg.get_tile(level=dzg.level_count-1, address=(i, j)))
                Image.fromarray(img).save(base_path + '{}_{}_{}.png'.format(path.split('.')[0].split('\\')[-1], i, j))
            except Exception:
                print("Error ----- > [{},{}]".format(i, j))
                continue


filename = 'D:\\Dataset\\Ovarian\\1427159G-Y.svs'
ef_base = 'D:\\Dataset\\Ovarian_After_Split\\Efficient\\'
in_base = 'D:\\Dataset\\Ovarian_After_Split\\Invalid\\'

# split_image(filename)
#
p = ef_base + '2-P001224800L_21_23.png'
p_2 = ef_base + '2-P001224800L_21_22.png'
test = Image.open(p)
test_2 = Image.open(p_2)
test = np.array(test.convert('L'))
test_2 = np.array(test_2.convert('L'), dtype=np.int64)

percent = np.sum(test <= 50) / test.size
percent_2 = np.sum(test_2 <= 50) / test_2.size

max(test_2[-1, :])
min(test_2[-1, :])

np.median(test_2[999, :])
np.min(test_2[999, :])
np.mean(test_2)

pe = np.sum((test_2 >= 41) & (test_2 <= 130)) / test.size
pe_1 = np.sum((test >= 41) & (test <= 130)) / test.size


percent_3 = np.sum(test >= 230) / test.size
percent_3 = np.sum(test_2 >= 230) / test_2.size
# #
# percent >= 0.012833
# percent_2 >= 0.01



path_base = 'D:\\Dataset\\Ovarian\\'
# path_list = os.listdir(in_base)


def filter_image(path):
    path_list = os.listdir(path)
    for i in tqdm(range(len(path_list))):
        file = path + path_list[i]
        try:
            img = Image.open(file)
            img = np.array(img.convert('L'))
            percent = np.sum(img >= 230) / img.size
            percent_2 = np.sum(img <= 50) / img.size
            pe_1 = np.sum((img >= 41) & (img <= 130)) / img.size
            if percent >= 0.87:
                os.remove(file)
                continue

            elif percent_2 >= 0.01:
                os.remove(file)
                continue

            elif pe_1 < 0.1:
                os.remove(file)
                continue

        except Exception:
            print("Error ----> {}".format(file))
            continue


filter_image(ef_base)


def preprocessing():
    data_list = os.listdir(path_base)[-5: ]
    for d in data_list:
        com_path = path_base + d
        com_list = os.listdir(com_path)

        for c in com_list:
            file = com_path + '\\' + c
            if 'in' in d:
                split_image(file, in_base)

            elif 'e' in d:
                split_image(file, ef_base)


# preprocessing()



# svs = openslide.OpenSlide('D:\Dataset\Ovarian\in5\\1707649A.svs')
# # Get the DeepZoomGenerator object to obtain the multiple level revolution radio
# dzg = DeepZoomGenerator(svs, tile_size=1000, limit_bounds=True, overlap=0)
# dzg.level_count-1
#
# img = np.array(dzg.get_tile(level=dzg.level_count-1, address=(0, 1)))
#
# data_list = os.listdir(path_base)[-5: ]