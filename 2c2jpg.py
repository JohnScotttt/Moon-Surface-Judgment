from Jtools import *

def convert_2C_to_jpg(floder_path):
    for file_name in listdir(floder_path):
        file_path = join(floder_path, file_name)
        if isfile(file_path):
            if file_name.endswith('.2C'):
                f = open(file_path, 'rb')
                data = f.read()
                byte_array = bytearray(data[len(data)-2352 * 1728 * 3:])
                numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((1728, 2352, 3))
                numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
                imwrite(file_path[:-2] + 'jpg', numpy_array)
                numpy_array = resize(numpy_array, (2352 // 2, 1728 // 2))
                f.close()

if __name__ == '__main__':
    floder_path = cmd.argv[1]
    convert_2C_to_jpg(floder_path)