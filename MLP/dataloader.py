from Jtools import *

def image_data_to_txt(data_path):
    images_path = join(data_path, 'images')
    masks_path = join(data_path, 'masks')
    image_list = []
    mask_list = []
    confirm(join(data_path, 'txt_data'))
    f = open(join(data_path, 'txt_data', 'total.txt'), 'w')
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

class txtDataset(Dataset):
    def __init__(self, txt):
        fh = open(txt, 'r')
        pixels = [] 
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('\t')
            pixels.append((float(words[0]), float(words[1]), float(words[2]), int(words[3])))
        self.pixels = pixels
        fh.close()

    def __getitem__(self, index):
        pixel = torch.FloatTensor([self.pixels[index][0], self.pixels[index][1], self.pixels[index][2]])
        label = self.pixels[index][3]
        return pixel, label

    def __len__(self):
        return len(self.pixels)

if __name__ == '__main__':
    image_data_to_txt('../data')
    shuffle_split('../data/txt_data/total.txt', '../data/txt_data/train.txt', '../data/txt_data/val.txt')