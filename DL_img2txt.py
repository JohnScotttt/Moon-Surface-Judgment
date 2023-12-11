from Jtools import *

def image_data_to_txt(input_path, output_path):
    path = ['images', 'masks']
    refresh(output_path)
    f = open(join(output_path, 'total.txt'), 'w')
    image_list = listdir(join(input_path, path[0]))
    mask_list = listdir(join(input_path, path[1]))
    for image in tqdm(image_list):
        if image[:-4]+".png" in mask_list:
            image_path = join(input_path, path[0], image)
            mask_path = join(input_path, path[1], image[:-4]+".png")
            f.write(image_path + '\t' + mask_path + '\n')
    f.close()

def date_clean(input_file, output_file):
    with open(input_file, 'r') as f:
        records = f.readlines()
        with open(output_file, 'w') as fc:
            for record in tqdm(records):
                record = record.strip('\n')
                record = record.rstrip()
                words = record.split('\t')
                if words[0].split("\\")[-1][:-4]+'.png' == words[1].split("\\")[-1]:
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
    image_data_to_txt('D:/repos/Moon-Surface-Judgment/data', 'D:/repos/Moon-Surface-Judgment/data/noslicing_DL_txt_data')
    date_clean('data/noslicing_DL_txt_data/total.txt', 'data/noslicing_DL_txt_data/total_c.txt')
    shuffle_split('data/noslicing_DL_txt_data/total_c.txt', 'data/noslicing_DL_txt_data/train.txt', 'data/noslicing_DL_txt_data/val.txt')