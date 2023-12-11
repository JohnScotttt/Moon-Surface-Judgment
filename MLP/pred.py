import sys
sys.path.append('..')

from Jtools import *

image_path = cmd.argv[1]
test_image = imread(image_path)
height, width, _ = test_image.shape

model = MLPNet()
model.load_state_dict(torch.load('output/params_117.pth'))
model.eval()

pred_image = np.zeros((height, width, 3))

for i in tqdm(range(height)):
    for j in tqdm(range(width), leave=False):
        out = model(torch.tensor(test_image[i][j]).float())
        if torch.max(out, 0)[1] == 1:
            pred_image[i][j] = [1, 0, 128]

imwrite('output/pred_' + image_path.split('/')[-1], pred_image)