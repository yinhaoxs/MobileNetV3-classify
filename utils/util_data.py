
m PIL import Image
import os, sys
import random
import shutil


## 删除无效的图片
def drop_invalid_image(path, des_path):
    # path = '/data2/haoyin/data/test/k/'  # 表示需要命名处理的文件夹
    # pathnew = '/data2/haoyin/k_error/'
    filelist = os.listdir(path)  # 获取文件路径
    i = 0
    for item in filelist:
        i = i + 1
        if i % 5 == 0:
            print(i)
        if item == '.DS_Store':
            continue
        imgpath = path + item
        imgpathnew = des_path + item

        try:
            img = Image.open(imgpath)
            h = img.height
            w = img.width
            if (h > 5 * w) or (w > 5 * h):
                os.remove(imgpath)
                continue
            else:
                shutil.move(imgpath, imgpathnew)

        except(OSError, NameError):
            print('OSError, Path:', imgpath)
            shutil.move(imgpath, imgpathnew)


## 移动文件到另一个文件夹
def change_move_image(path, des_path):
    # path = '/data2/haoyin/data/train/k1/'  # 表示需要命名处理的文件夹
    # des_path = "/data2/haoyin/data/train/k/"
    # 图片文件改名
    num=0
    for file in os.listdir(path):
       path_dir = os.path.join(path+os.sep, file)
       os.rename(path_dir, os.path.join(path+os.sep, "{}_1.jpg".format(num)))
       num+=1
    # 将改名后的文件移动到目标文件夹
    for file in os.listdir(path):
        path_dir = os.path.join(path+os.sep, file)
        shutil.move(path_dir, des_path)


## 对样本进行划分：训练集与测试集
def sample_shuffle_image(path, des_path):
    for img_dir in os.listdir(path):
        if img_dir.endswith(".tar"):
            continue
        elif img_dir.startswith("."):
            continue
        else:
            aim_path = os.path.join(path + os.sep, img_dir)
            pathDir = os.listdir(aim_path)
            sample = random.sample(pathDir, int(len(pathDir)*0.2))

            print(sample)
            for name in sample:
                if name.startswith("."):
                    continue
                else:
                    shutil.move(aim_path + "/" + name, des_path)


## 对log日志进行信息的剔除
def remove_redundant(log_1, log_2):
    with open(log_1, 'r') as f1, open(log_2, 'w') as f2:
        for line in f1.readlines():
            line = line.strip()
            if "Train Loss" in line:
                f2.write(line + '\n')
            elif "Test Loss" in line:
                f2.write(line + '\n')
            elif "epoch" in line:
                f2.write(line + '\n')
            else:
                print("drop redundant information!")


## 画图train loss与test loss曲线
def draw_loss_curve(train_acc, valid_acc):
    x_length = len(train_acc)
    x = range(0, x_length, 1)
    plt.plot(x, train_acc, marker='.', color='r', label=u'train_acc')
    plt.plot(x, valid_acc, marker='*', color='r', label=u'valid_acc')
    plt.legend() # legend effective
    plt.xticks(x, rotation=45)
    plt.ylim(0.1, 1)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u'EPOCHES')
    plt.ylabel(u'ACC')
    plt.title(u'ACC')
    plt.show()



if __name__ == '__main__':
    path = '/data2/haoyin/data/train/k1/'  # 表示需要命名处理的文件夹
    des_path = "/data2/haoyin/data/train/k/"
    drop_invalid_image(path, des_path)
    change_move_image(path, des_path)
    sample_shuffle_image(path, des_path)
    # remove_redundant(log_1, log_2)




