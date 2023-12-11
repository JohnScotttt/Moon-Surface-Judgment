from Jtools import *

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

class imgDataset(Dataset):
    def __init__(self, txt, transform=None):
        fh = open(txt, 'r')
        images = []
        masks = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('\t')
            images.append(words[0])
            masks.append(words[1])
        self.images = images
        self.masks = masks
        self.transform = transform
        fh.close()

    def __getitem__(self, index):
        image = imread(self.images[index])
        mask = imread(self.masks[index])
        mask = np.squeeze(mask[:, :, :1], axis=2)
        if self.transform is not None:
            image = self.transform(image)
            mask = tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.images)