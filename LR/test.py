from net import LRNet
from Jtools import *

image_path = cmd.argv[1]
test_image = imread(image_path)
height, width, _ = test_image.shape

model = LRNet()
model.load_state_dict(torch.load('output/params_150.pth'))
model.eval()

pred_image = np.zeros((height, width, 3))

# def predict_pixel(i, j):
#     out = model(torch.tensor(test_image[i][j]).float())
#     if torch.max(out, 0)[1] == 1:
#         pred_image[i][j] = [1, 0, 128]

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(predict_pixel, i, j) for i in tqdm(range(height)) for j in tqdm(range(width), leave=False)]
#     for future in tqdm(concurrent.futures.as_completed(futures), total=height*width):
#         pass

# imwrite('output/pred_' + image_path.split('/')[-1], pred_image)


for i in tqdm(range(height)):
    for j in tqdm(range(width), leave=False):
        out = model(torch.tensor(test_image[i][j]).float())
        if torch.max(out, 0)[1] == 1:
            pred_image[i][j] = [1, 0, 128]

imwrite('output/pred_' + image_path.split('/')[-1], pred_image)