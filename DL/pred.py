import sys
sys.path.append('..')

from Jtools import *

image_path = cmd.argv[1]
test_image = imread(image_path)
height, width, _ = test_image.shape

model = UNet()
model.load_state_dict(torch.load('output/params_182.pth'))
model.eval().cuda()

out = model(torch.unsqueeze(transforms.ToTensor()(test_image), axis=0).cuda())
pred = torch.max(out, 1)[1]

pred_image1 = torch.unsqueeze(pred[0], axis=2)
pred_image2 = torch.zeros(height, width, 1).cuda()
pred_image3 = torch.unsqueeze(pred[0]*128, axis=2)
pred_image = torch.cat((pred_image1, pred_image2, pred_image3), dim=2)
pred_image = pred_image.cpu().numpy().astype(np.uint8)
imwrite('output/pred_' + image_path.split('/')[-1], pred_image)