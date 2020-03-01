import os

def generate(dir, label):
    files = os.listdir(dir)
    # files.sort()
    print('****************')
    print('input :', dir)
    print('start...')
    listText = open(dir + '\\' + 'test_num.txt', 'w')
    for file in files:
        (filename, extension) = os.path.splitext(file)
        if extension == '.txt':
            continue
        name = filename + '\t' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()
    print('down!')
    print('****************')


if __name__ == '__main__':
    generate('C:\\Users\\67304\\Desktop\\Python Studio\\ResNet-18实现CIFAR-10图像分类\\Numpy-Implementation-of-ResNet-master\\test\\0', 0)