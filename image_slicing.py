from Jtools import *

def image_slicing(size, input_path, output_path):
    for image in tqdm(listdir(input_path)):
        img = imread(join(input_path, image))
        height = img.shape[0]
        width = img.shape[1]
        for i in range(0, height, size):
            for j in range(0, width, size):
                if i + size > height:
                    i = height - size
                if j + size > width:
                    j = width - size
                slice = img[i:i + size, j:j + size]
                imwrite(join(output_path, image[:-4] + '_' + str(i) + '_' + str(j) + '.png'), slice)
        
if __name__ == '__main__':
    image_input_path = "D:/repos/Moon-Surface-Judgment/data/images"
    image_output_path = "D:/repos/Moon-Surface-Judgment/data/slicing/images"
    mask_input_path = "D:/repos/Moon-Surface-Judgment/data/masks"
    mask_output_path = "D:/repos/Moon-Surface-Judgment/data/slicing/masks"
    refresh(image_output_path)
    refresh(mask_output_path)
    image_slicing(256, image_input_path, image_output_path)
    image_slicing(256, mask_input_path, mask_output_path)