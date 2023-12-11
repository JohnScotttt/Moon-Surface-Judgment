from Jtools import *

def image_data_to_txt(input_path, output_path):
    images_path = join(input_path, 'images')
    masks_path = join(input_path, 'masks')
    image_list = []
    mask_list = []
    refresh(output_path)
    f = open(join(output_path, 'total.txt'), 'w')
    for image in listdir(images_path):
        image_list.append(image)
    for mask in listdir(masks_path):
        mask_list.append(mask)
    for image in tqdm(image_list):
        if image[:-4]+".png" in mask_list:
            image_data = imread(join(images_path, image)).reshape(-1, 3)
            mask_data = imread(join(masks_path, image[:-4]+".png")).reshape(-1, 3)
            for i in tqdm(range(image_data.shape[0]),leave=False):
                f.write(str(image_data[i][0]) + '\t' + str(image_data[i][1]) + '\t' + str(image_data[i][2]) + '\t' + str(mask_data[i][0]) + '\n')
    f.close()

def date_clean(input_file, output_file):
    with open(input_file, 'r') as f:
        records = f.readlines()
    with open(output_file, 'w') as fc:
        for record in tqdm(records):
            record = record.strip('\n')
            record = record.rstrip()
            words = record.split('\t')
            if words[3] == '0':
                if rd.random() < 0.1:
                    fc.write(record + '\n')
            else:
                fc.write(record + '\n')

def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    rd.shuffle(records)
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as ft:
        ft.writelines(records[0:trainNum])
    with open(valFile, 'w') as fv:
        fv.writelines(records[trainNum:])

if __name__ == '__main__':
    image_data_to_txt('data', 'data/ML_txt_data')
    date_clean('data/ML_txt_data/total.txt', 'data/ML_txt_data/total_c.txt')
    shuffle_split('data/ML_txt_data/total_c.txt', 'data/ML_txt_data/train.txt', 'data/ML_txt_data/val.txt')