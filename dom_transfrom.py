from Jtools import *

def dom_transform(image_path):
    src = imread(image_path)

    srcWidth = src.shape[1]
    srcHeight = src.shape[0]
    dist_coefs = np.array([0, 0, 0, 0, 0], dtype = np.float64)
    camera_matrix = np.array([[srcWidth * 0.5, 0, srcWidth / 2], [0, srcWidth * 0.5, srcHeight / 2], [0, 0, 1]],
                            dtype = np.float64)

    newWidth = 1280  # 新图像宽
    newHeight = 720  # 新图像高
    # 新相机内参，自己设定
    newCam = np.array([[newWidth * 0.15, 0, newWidth / 2], [0, newWidth * 0.15, newHeight / 2], [0, 0, 1]])
    invNewCam = np.linalg.inv(newCam)  # 内参逆矩阵
    map_x = np.zeros((newHeight, newWidth), dtype=np.float32)
    map_y = np.zeros((newHeight, newWidth), dtype=np.float32)

    pitch = 50 * 3.14 / 180
    R = np.array([[1, 0, 0], [0, math.sin(pitch), math.cos(pitch)], [0, -math.cos(pitch), math.sin(pitch)]])
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            ray = np.dot(invNewCam, np.array([j, i, 1]).T)  # 像素转换为入射光线
            rr = np.dot(R, ray)  # 乘以旋转矩阵
            # 光线投影到像素点
            point, _ = cv2.projectPoints(rr, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), camera_matrix,
                                        dist_coefs)
            map_x[i, j] = point[0][0][0]
            map_y[i, j] = point[0][0][1]
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    imwrite("dom.png", dst)

if __name__ == '__main__':
    image_path = "D:/repos/Moon-Surface-Judgment/DL/output/pred_CE3_BMYK_PCAML-C-006_SCI_N_20140112131703_20140112131703_0007_A.jpg"
    dom_transform(image_path)