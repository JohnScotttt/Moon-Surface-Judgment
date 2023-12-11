from Jtools import *

def json_to_mask(label_name):
    with open(join(root_dir, 'annotations', label_name + '.json'), 'r') as obj:
        dict = load_json(obj)
    img = imread(join(root_dir, 'images', label_name + '.jpg'))
    black_img = nzeros(img.shape)
    for label in dict['shapes']:
        points = narray(label['points'], dtype=np.int32)
        cv2.polylines(black_img, [points], isClosed=True, color=fill_color, thickness=1)
        cv2.fillPoly(black_img, [points], color=fill_color)

    imwrite(join(root_dir, 'masks', label_name + '.png'), black_img)

if __name__ == '__main__':
    fill_color = (1, 0, 128)
    root_dir = "data"
    refresh(join(root_dir, 'masks'))
    for i in tqdm(listdir(join(root_dir, "annotations"))):
        label_name = i[:-5]
        json_to_mask(label_name)