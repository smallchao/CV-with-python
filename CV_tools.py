#-*- coding:utf-8 -*-
import os,requests,math,jieba
from io import BytesIO
import numpy as np
import mahotas as mh
from mahotas.features import surf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from scipy.ndimage import filters
from scipy.signal import convolve2d
import cv2 as cv
from wordcloud import WordCloud,ImageColorGenerator
import warnings; warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

#========================================================
#  加载图片
#========================================================
def read_image(path, mode='RGB'):
    '''
    加载本地单张图片
    INPUT  -> 单张图片路径, 特殊模式
    OUTPUT -> 单张图文件
    '''
    pil_im = Image.open(path)
    if mode == '1':
        # 非黑即白模式: 0表示黑, 255表示白
        return pil_im.convert('1')  
    elif mode == 'L':
        # 灰色模式: 0表示黑，255表示白，其他数字表示不同的灰度。转换算法 L = R * 299/1000 + G * 587/1000+ B * 114/1000
        return pil_im.convert('L')
    elif mode == 'RGB':
        return pil_im.convert('RGB')

def read_net_image(url, mode='RGB'):
    '''
    加载网络单张图片
    INPUT  -> 单张图片网址, 特殊模式
    OUTPUT -> 单张图文件
    '''
    pil_im = Image.open(BytesIO(requests.get(url).content))
    if mode == '1':
        return pil_im.convert('1')
    elif mode == 'L':
        return pil_im.convert('L')
    elif mode == 'RGB':
        return pil_im

def show_color(im):
    '''
    绘制颜色通道
    INPUT  -> 单张图片
    '''
    # im = mpimg.imread('wa_state_highway.jpg')
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.set_title('R channel')
    ax1.imshow(r, cmap='gray')
    ax2.set_title('G channel')
    ax2.imshow(g, cmap='gray')
    ax3.set_title('B channel')
    ax3.imshow(b, cmap='gray')

def image_process_batch(path):
    '''
    图像批量处理
    INPUT  -> 图像所在文件夹
    '''
    files = os.listdir(path) # 得到文件夹下的所有文件名称
    # 遍历文件夹
    prefix = path+'/'
    for file in files: 
        if file.endswith('.jpg'):
            pass
            # image_process(prefix+file)

#========================================================
#  图像数组化与还原
#========================================================

def image_to_array(pil_im):
    '''
    图片转化为数组
    INPUT  -> 单张图文件
    OUTPUT -> 数组
    '''
    return np.array(pil_im, 'f')

def array_to_image(image_arr):
    '''
    数组还原为图片
    INPUT  -> 数组
    OUTPUT -> 单张图文件
    '''
    if len(image_arr.shape) == 3:  # 格式为(height(rows), weight(colums), 3)
        r = Image.fromarray(np.uint8(image_arr[:,:,0]))
        g = Image.fromarray(np.uint8(image_arr[:,:,1]))
        b = Image.fromarray(np.uint8(image_arr[:,:,2]))
        image = Image.merge("RGB", (r, g, b))
        return image        
    elif len(image_arr.shape) == 2:  # 格式为(height(rows), weight(colums))
        return Image.fromarray(np.uint8(image_arr))

#========================================================
#  图像预处理之滤波:去噪和边缘检测
#========================================================

def image_saltnoise(pil_im, snr):
    '''
    给图像添加椒盐噪声
    INPUT  -> 单张图像文件
    OUTPUT -> 添加椒盐噪点后的图像文件
    '''
    image_arr = image_to_array(pil_im)
    # 指定信噪比
    SNR = snr
    # 获取总共像素个数
    size = image_arr.shape[0] * image_arr.shape[1]
    # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
    noiseSize = int(size * (1 - float(SNR)))
    # 对这些点加噪声
    for k in range(0, noiseSize):
        # 随机获取 某个点
        xi = int(np.random.uniform(0, image_arr.shape[1]))
        xj = int(np.random.uniform(0, image_arr.shape[0]))
        # 增加噪声
        if len(image_arr.shape) == 2:
            image_arr[xj, xi] = 255
        elif len(image_arr.shape) == 3:
            image_arr[xj, xi] = 0
    return array_to_image(image_arr)

def image_medium_filter(image_arr, K=5):
    '''
    中值滤波
    中值平滑只对特别尖锐的信号平滑
    INPUT  -> 图像数组
    OUTPUT -> 去噪后的图像数组
    '''
    if len(image_arr.shape) == 3:
        for i in range(3):
           image_arr[:,:,i] = filters.median_filter(image_arr[:,:,i], K)
        return image_arr   
    elif len(image_arr.shape) == 2:
        image_arr = filters.medianBlur(image_arr, K)
        return image_arr

def image_gaussian_filter(image_arr, K=5):
    '''
    高斯滤波
    INPUT  -> 图像的数组, 卷积核
    OUTPUT -> 处理后的数组
    '''
    if len(image_arr.shape) == 3:
        for i in range(3):
           image_arr[:,:,i] = filters.gaussian_filter(image_arr[:,:,i], K)
        return image_arr   
    elif len(image_arr.shape) == 2:
        return array_to_image(filters.gaussian_filter(image_arr, K))

def image_edge_1(pil_im):
    '''
    边缘检测1
    不同级别高斯滤波作差放大并加到基准值上
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    image_arr = image_to_array(pil_im)
    gauss_out1 = filters.gaussian_filter(image_arr, sigma=1)
    gauss_out3 = filters.gaussian_filter(image_arr, sigma=3)
    sharp = gauss_out3 + 6*(gauss_out3-gauss_out1)
    return array_to_image(sharp)

def image_edge_2(pil_im):
    '''
    边缘检测2
    高反差保留 = 原图 - 高斯模糊图, 然后加一个灰度底
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    image_arr = image_to_array(pil_im)
    gauss_out = filters.gaussian_filter(image_arr, sigma=5)
    img_out = image_arr - gauss_out + 128.0
    return array_to_image(img_out)

#========================================================
#  图像预处理之增强：直方图均衡化
#  针对曝光过度或者逆光拍摄, 增强图像细节
#========================================================
def image_grayhist(pil_im):
    '''
    计算灰度直方图
    INPUT  -> 单张图文件
    OUTPUT -> 灰度直方图
    '''
    image_arr = image_to_array(pil_im)
    w, h = image_arr.shape
    grayHist = np.zeros([256], np.uint8)
    for i in range(w):
        for j in range(h):
            grayHist[int(image_arr[i][j])] += 1
    return grayHist

def image_histeq(pil_im):
    '''
    直方图均衡化
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    # 计算图像的直方图
    image_arr = image_to_array(pil_im)
    imhist, bins = np.histogram(image_arr.flatten(), 256, normed=True)
    cdf = imhist.cumsum()   # 累计分布函数
    cdf = 255*cdf/cdf[-1]   # 归一化
    # 使用累计分布函数的线性插值计算新的像素值
    image_arr2 = np.interp(image_arr.flatten(), bins[:-1], cdf)
    return array_to_image(image_arr2.reshape(image_arr.shape))

#========================================================
#  图像预处理之增强：对数增强
#  对数变换对于图像对比度偏低，并且整体亮度值偏低（相机光照不足）情况下的图像增强效果明显
#========================================================

def image_log_transform(pil_im):
    '''
    图像对数增强
    公式为： S = c*log(r+1),其中s和r是输出和输入图片的像素值,c是常数(c=255/log(256))
    扩展和增强低灰度部分,压缩高灰度部分的值, 实现整个画面的亮度增大,
    
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    image_arr = image_to_array(pil_im)
    if len(image_arr.shape) == 3:
        for i in range(3):
            image_arr[:,:,i] = 255/np.log(255+1)*np.log(1+image_arr[:,:,i])
        return array_to_image(image_arr)
    elif len(image_arr.shape) == 2:
        # 值1被添加到输入图片的每个像素值,如果图片中的像素强度为0,则log(0)等于无穷大,添加1的作用是使最小值至少为1。
        # image_arr = 255/np.log(np.max(image_arr)+1)*np.log(1+image_arr)
        image_arr = 255/np.log(255+1)*np.log(1+image_arr)
        return array_to_image(image_arr)

#========================================================
#  图像预处理之增强：伽马校正
#  伽马变换对于图像对比度偏低，并且整体亮度值偏高（对于于相机过曝）情况下的图像增强效果明显
#========================================================

def image_gamma_correction(pil_im, gamma):
    '''
    伽马校正
    V1 = V0 **(1/G),其中V0是输入图像,V1是输出图像,G是伽马值
    V0图片像素强度必须从0-255到0-1.0的范围缩放,V1输出图像恢复0-255
    Gamma值<1会将图像移向光谱的较暗端,Gamma值>1将使图像显得更亮
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    image_arr = image_to_array(pil_im)
    if len(image_arr.shape) == 3:
        for i in range(3):
            image_arr[:,:,i] = ((image_arr[:,:,i]/255) ** (1/gamma))
        return array_to_image(image_arr*255)
    elif len(image_arr.shape) == 2:
        image_arr = ((image_arr/255) ** (1/gamma))
        return array_to_image(image_arr*255)

#========================================================
#  图像预处理之增强：拉普拉斯算子
#  拉普拉斯算子是一种微分算子,强调的是图像中灰度的突变,实现锐化处理的效果
#========================================================

def image_laplace_sharp(pil_im):
    '''
    拉普拉斯算子增强
    INPUT  -> 单张图文件
    OUTPUT -> 处理后的图文件
    '''
    image_arr = image_to_array(pil_im)
    dst_arr = np.zeros_like(image_arr)
    # 卷积核-拉普拉斯算子
    laplace_operator = np.array([[0, -1, 0], 
                                [-1, 5, -1], 
                                [0, -1, 0]])
    if len(image_arr.shape) == 3:
        for i in range(3):
            dst_arr[:,:,i] = convolve2d(image_arr[:,:,i], laplace_operator, mode="same")
    elif len(image_arr.shape) == 2:
        dst_arr = convolve2d(image_arr, laplace_operator, mode="same")
    dst_arr = image_arr + image_gaussian_filter(dst_arr, 5)
    dst_arr = dst_arr / 255.0
    # 饱和处理
    mask_1 = dst_arr  < 0 
    mask_2 = dst_arr  > 1
    dst_arr = dst_arr * (1-mask_1)
    dst_arr = dst_arr * (1-mask_2) + mask_2
    return array_to_image(dst_arr*255)

#========================================================
#  图像处理:裁剪、水印、画框
#========================================================
def image_clip(pil_im, x, y, width, height):
    '''
    从一张图片中截取某个区域
    INPUT  -> 单张图文件, 坐标点x, 坐标点y, 目标宽度, 目标高度
    OUTPUT -> 截取后的图片
    '''
    im_w, im_h = pil_im.size
    x2 = x + width
    y2 = y + height
    if x + width > im_w:
        x2 = im_w
    if y + height > im_h:
        y2 = im_h
    box = (x, y, x2, y2)
    region = pil_im.crop(box)
    return region

def image_watermark_logo(ori_im, mark_im):
    '''
    给一张图片加上图片水印
    INPUT  -> 原始图片, 水印图片, 透明度
    OUTPUT -> 处理后的图片
    '''
    ori_w, ori_h = ori_im.size
    mark_w, mark_h = mark_im.size
    # 图层
    dst_im = Image.new('RGBA', ori_im.size, (255,255,255,0))  # 设置图片尺寸和透明度
    dst_im.paste(mark_im, (ori_w-mark_w, ori_h-mark_h))
    # 覆盖
    dst_im = Image.composite(dst_im, ori_im, dst_im)
    return dst_im

def image_watermark_text(ori_im, fontpath, text=''):
    '''
    给一张图片加上文字水印
    INPUT  -> 原始图片, 字体路径, 文本
    OUTPUT -> 处理后的图片
    '''
    ori_im = ori_im.convert('RGBA')
    ori_w, ori_h = ori_im.size
    # 文本遮罩层
    text_overlay = Image.new('RGBA', ori_im.size, (255,255,255,0))
    image_draw = ImageDraw.Draw(text_overlay)
    # 获取文本大小
    fnt = ImageFont.truetype(fontpath, 20)
    text_size_x, text_size_y = image_draw.textsize(text, font=fnt)
    # 设置文本位置
    text_xy = (ori_w-text_size_x, ori_h-text_size_y)
    # 设置文本颜色和透明度
    image_draw.text(text_xy, text, font=fnt, fill=(255,255,255,50))
    # 覆盖
    dst_im = Image.alpha_composite(ori_im, text_overlay)
    return dst_im

def image_tag(pil_im, x1, y1, x2, y2):
    '''
    在图片上标记矩形框
    INPUT  -> 单张图文件, 对角点坐标(PIL使用笛卡尔像素坐标系统，坐标(0，0)位于左上角)
    OUTPUT -> 绘制后的图文件
    '''
    canvas = ImageDraw.Draw(pil_im)
    canvas.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], fill='red')
    return canvas

#========================================================
#  图像基本变换:刚体变换和相似变换
#  包含了旋转、放缩、拼接
#========================================================

def image_rotate(pil_im, angle, clockwise=False):
    '''
    使一张图片旋转
    INPUT  -> 单张图文件, 旋转角度, 旋转方向(默认逆时针)
    OUTPUT -> 旋转后的图文件
    '''
    # 转换为有alpha层
    temp_im = pil_im.convert('RGBA')
    if clockwise:
        angle = 360 - int(angle)
        temp_im = temp_im.rotate(angle)
    else: 
        temp_im = temp_im.rotate(angle)
    # 创建一个与旋转图像大小相同的白色图像
    fff = Image.new('RGBA', temp_im.size, (255,)*4)
    # 使用alpha层的temp_im作为掩码创建一个复合图像
    out = Image.composite(temp_im, fff, temp_im)
    # 去掉alpha层
    out = out.convert(pil_im.mode)
    return out

def image_resize1(pil_im, dst_w, dst_h):
    '''
    使一张图片变换尺寸(非等比例)(指定宽高)
    INPUT  -> 单张图文件, 目标宽度, 目标高度
    OUTPUT -> 处理后的图片
    '''
    return pil_im.resize((dst_w, dst_h))

def image_resize_proportionally1(pil_im, c):
    '''
    使一张图片变换尺寸(等比例)(倍数缩放)
    INPUT  -> 单张图文件, 目标宽度, 目标高度
    OUTPUT -> 处理后的图片
    '''
    height, width = pil_im.size
    dst_w = height*c
    dst_h = width*c
    return pil_im.resize((dst_w, dst_h))

def image_resize_proportionally2(pil_im, dst_w, dst_h):
    '''
    使一张图片变换尺寸(等比例)
    INPUT  -> 单张图文件, 目标宽度, 目标高度
    OUTPUT -> 处理后的图片
    '''
    ori_w, ori_h = pil_im.size
    widthRatio = heightRatio = None
    ratio = 1
    if (ori_w and ori_w > dst_w) or (ori_w and ori_h > dst_h):
        if (ori_w > dst_w):
            widthRatio = float(dst_w) / ori_w # 获取宽度缩放比例
        if (ori_h > dst_h):
            heightRatio = float(dst_h) / ori_h # 获取高度缩放比例
        if widthRatio and heightRatio:
            if widthRatio < heightRatio:
                ratio = widthRatio
            else:
                ratio = heightRatio
        if widthRatio and not heightRatio:
            ratio = widthRatio
        if heightRatio and not widthRatio:
            ratio = heightRatio
        newWidth = int(ori_w * ratio)
        newHeight = int(ori_h * ratio)
    else:
        newWidth = ori_w
        newHeight = ori_h
    return pil_im.resize((newWidth,newHeight), Image.ANTIALIAS)

def image_append(pil_im1, pil_im2):
    '''
    将两幅图像并排拼接成的一幅新图像      有问题！！！
    INPUT  -> 图像1(灰度), 图像2(灰度)
    OUTPUT -> 新图像
    '''
    # 选取具有最少行数的图像，然后填充足够的空行
    
    image_arr1 = image_to_array(pil_im1)
    image_arr2 = image_to_array(pil_im2)
    
    row1 = image_arr1.shape[0]
    row2 = image_arr2.shape[0]
    col1 = image_arr1.shape[1]
    col2 = image_arr2.shape[1]

    if row1 < row2:
        image_arr1 = np.concatenate((image_arr1, np.zeros((row2-row1, col1))), axis=0)
    elif row1 > row2:
        image_arr2 = np.concatenate((image_arr2, np.zeros((row1-row2, col2))), axis=0)
    return array_to_image(np.concatenate((image_arr1, image_arr2), axis=1))

#========================================================
#  图像特征描述子
#  SIFT特征对于尺度、旋转、亮度都具有不变性,SURF是对SIFT的升级
#========================================================
def image_SURF_kp(pil_im):
    '''
    提取图像的SURF描述子
    INPUT  -> 单张图文件
    OUTPUT -> 标识特征点的图片, SURF特征点
    '''
    pil_im = pil_im.convert('L')
    image_arr = image_to_array(pil_im)
    spoints = surf.surf(image_arr, 4, 6)
    kp_image_arr = surf.show_surf(image_arr, spoints)
    kp_image = array_to_image(kp_image_arr)
    return kp_image, spoints

def analysis_SURF_kp(kp):
    '''
    分析SURF描述子
    INPUT  -> SURF特征点
    OUTPUT -> 特征位置, 其他内容
    '''
    return kp[:,:4], kp[:,4:]

def plot_image_feature(pil_im, kploc, circle=False):
    '''
    绘制单张图片的特征点
    INPUT  -> 单张图文件, 特征位置, 是否用圆圈表示
    '''
    def draw_circle(c, r):
        t = np.arange(0, 1.01, 0.01)*2*math.pi
        x = r*np.sin(t)+c[1]
        y = r*np.cos(t)+c[0]
        plt.plot(x, y, 'b', linewidth=2)
    
    if circle:
        for p in kploc:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(kploc[:,1], kploc[:,0], 'ob')

    plt.imshow(pil_im)
    plt.title('The image kp')
    plt.axis('off')
    plt.show()

#========================================================
#  图像描述子匹配 (有问题)
# https://blog.csdn.net/zhuxiaoyang2000/article/details/53930610
#========================================================
def match_kp(kp1, kp2):
    '''
    匹配两张图片的描述子
    INPUT  -> 第一张图中描述子, 第二张图中描述子
    '''
    kp1 = np.array([d/np.linalg.norm(d) for d in kp1])
    kp2 = np.array([d/np.linalg.norm(d) for d in kp2])

    dist_ratio = 0.6
    kp1_size = kp1.shape
    matchscores = np.zeros((kp1_size[0], 1), 'int')
    kp2t = kp2.T #预先计算矩阵转置

    for i in range(kp1_size[0]):
        dotprods = np.dot(kp1[i,:], kp2t) #向量点乘
        dotprods = 0.9999*dotprods
        # 反余弦和反排序，返回第二幅图像中特征的索引
        indx = np.argsort(np.arccos(dotprods))

        #检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores

def get_good_match(kp1, kp2):
    '''
    双向对称匹配,去除错误匹配
    INPUT  -> 第一张图中描述子, 第二张图中描述子
    '''
    matches_12 = match_kp(kp1, kp2)
    matches_21 = match_kp(kp2, kp1)

    ndx_12 = matches_12.nonzero()[0]

    # 去除不对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12

def plot_good_match(pil_im1, pil_im2, kp1, kp2, good_match):
    '''
    绘制单张图片的特征点
    INPUT  -> 图像1, 图像2, 图像1的描述子, 图像2的描述子, 描述子匹配
    '''
    im3 = image_append(pil_im1, pil_im2)
    image_arr1 = image_to_array(pil_im1)
    image_arr2 = image_to_array(pil_im2)
    kploc1, oth1 = analysis_SURF_kp(kp1)
    kploc2, oth2 = analysis_SURF_kp(kp2)
    cols1 = image_arr1.shape[1]
    for i in range(len(good_match)):
        if good_match[i]>0:
            plt.plot([kploc1[i,0], kploc2[good_match[i,0],0]+cols1], [kploc1[i,1], kploc2[good_match[i,0], 1]], 'c')
    plt.imshow(im3)
    plt.axis('off')
    plt.show()

#========================================================
#  图像高级变换:仿射变换、相似变换、投影变换
#  包含图像扭曲、图像配准、全景图
#========================================================


def pointcloud2image(point_cloud):
    x_size = 640
    y_size = 640
    x_range = 60.0
    y_range = 60.0
    grid_size = np.array([2 * x_range / x_size, 2 * y_range / y_size])
    image_size = np.array([x_size, y_size])
    # [0, 2*range)
    shifted_coord = point_cloud[:, :2] + np.array([x_range, y_range])
    # image index
    index = np.floor(shifted_coord / grid_size).astype(np.int)
    # choose illegal index
    bound_x = np.logical_and(index[:, 0] >= 0, index[:, 0] < image_size[0])
    bound_y = np.logical_and(index[:, 1] >= 0, index[:, 1] < image_size[1])
    bound_box = np.logical_and(bound_x, bound_y)
    index = index[bound_box]
    # show image
    image = np.zeros((640, 640), dtype=np.uint8)
    image[index[:, 0], index[:, 1]] = 255
    res = Image.fromarray(image)
    # rgb = Image.merge('RGB', (res, res, res))
    res.show()







 
#========================================================
#  图像生成
#========================================================
def image_wordcloud(text, pil_im):
    '''
    生成词云图
    INPUT  -> 文本, 图像文件
    '''
    # 从图像创建着色
    mask_image_arr = image_to_array(pil_im)
    image_colors = ImageColorGenerator(mask_image_arr)
    # 通过jieba分词进行分词并通过空格分隔
    wordlist_after_jieba = jieba.cut(text, cut_all = False)
    wordlist_space_split = " ".join(wordlist_after_jieba)
    # 生成词云
    cloud = WordCloud(font_path='C:/Users/Windows/fonts/simkai.ttf',  # 英文的不导入也并不影响，若是中文的或者其他字符需要选择合适的字体包
                      background_color='white',  # 设置背景颜色
                      mask=mask_image_arr,    # 设置掩膜,产生词云背景的区域
                      max_words=2000,    # 设置最大显示的字数
                      max_font_size=80,  # 设置字体最大值
                      random_state=40,    # 设置有多少种配色方案
                      margin=5).generate(wordlist_space_split)
    cloud.recolor(color_func=image_colors)
    # 保存图片
    cloud.to_file(os.path.join(FILE_DIR, 'my_wordcloud.png'))

def image_newyear(dd):
    '''
    生成春节倒计时
    INPUT  -> 剩余日期
    '''
    # 创建图像,设置图像大小及颜色
    im = Image.new('RGBA', (1000, 1800), (166, 12, 4, 255))
    draw = ImageDraw.Draw(im)
    # 设置本次使用的字体
    fontsFolder = 'C:UsersWindowsFonts'
    font1 = ImageFont.truetype('C:/Users/Windows/fonts/simkai.ttf', 420)  #华康俪金黑W8
    font2 = ImageFont.truetype(os.path.join(fontsFolder,'simkai.ttf'), 40)  #方正兰亭刊黑
    # 计算各文本的放置位置
    txtSize_1 = draw.textsize('距 离 除 夕 夜', font2)
    pos_x_1 = (1000 - txtSize_1[0]) / 2
    txtSize_2 = draw.textsize('还 有', font2)
    pos_x_2 = (1000 - txtSize_2[0]) / 2
    txtSize_3 = draw.textsize('天', font2)
    pos_x_3 = (1000 - txtSize_3[0]) / 2
    txtSize_4 = draw.textsize('不 是 年 味 越 来 越 少', font2)
    pos_x_4 = (1000 - txtSize_4[0]) / 2
    txtSize_5 = draw.textsize('而 是 我 们 都 长 大 了', font2)
    pos_x_5 = (1000 - txtSize_5[0]) / 2
    # 设置文本放置位置,居中
    draw.text((pos_x_1, 200), '距 离 除 夕 夜', fill=(217, 217, 217, 255), font=font2)
    draw.text((pos_x_2, 300), '还 有', fill=(217, 217, 217, 255), font=font2)
    draw.text((pos_x_3, 1050), '天', fill=(217, 217, 217, 255), font=font2)
    draw.text((pos_x_4, 1350), '不 是 年 味 越 来 越 少', fill=(137, 183, 109, 255), font=font2)
    draw.text((pos_x_5, 1440), '而 是 我 们 都 长 大 了', fill=(137, 183, 109, 255), font=font2)
    # 绘制线框
    draw.line([(20, 20), (980, 20), (980, 1780), (20, 1780), (20, 20)], fill=(217, 217, 217, 255), width=5)
    # 设置变化的文本属性
    txtSize_6 = draw.textsize(str(dd), font1)
    pos_x_6 = (1000 - txtSize_6[0]) / 2
    draw.text((pos_x_6, 500), str(dd), fill=(137, 183, 109, 255), font=font1)
    # 保存图像
    filename = '\day' + str(dd) + '.png'
    im.save(os.path.join(FILE_DIR, filename))

def image_sketch(pil_im, depths=10):
    '''
    生成素描图
    INPUT  -> 原始图片, 深度
    '''
    image_arr = image_to_array(pil_im.convert('L'))
    depth = depths  # 深度的取值范围(0-100),标准取10

    # 梯度的重构
    grad = np.gradient(image_arr)  # 取图像灰度的梯度值
    grad_x, grad_y = grad  # 分别取横纵图像梯度值
    grad_x = grad_x * depth / 100.  # 对grad_x值进行归一化
    grad_y = grad_y * depth / 100.  # 对grad_y值进行归一化
    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    # 设置光源
    vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
    vec_az = np.pi / 4.  # 光源的方位角度，弧度值
    dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对x 轴的影响
    dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对y 轴的影响
    dz = np.sin(vec_el)  # 光源对z 轴的影响

    image_arr = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 光源归一化
    image_arr = image_arr.clip(0, 255)
    im = array_to_image(image_arr)  # 重构图像
    im.save(os.path.join(FILE_DIR, 'sketch.jpg'))


def image_average_L(imlist):
    '''
    计算图像列表的平均图像(灰度)
    INPUT  -> 图像列表
    OUTPUT -> 一张平均图像(图像格式)
    '''
    skipped = 0
    # 打开第一幅图像并将其存入在浮点型数组中
    averageim = image_to_array(read_image(imlist[0], mode='L'))
    for imname in imlist[1:]:
        try:
            averageim += image_to_array(read_image(imname, mode='L'))
        except:
            print(imname+'....skipped')
            skipped += 1
    averageim /= (len(imlist)-skipped)
    return array_to_image(averageim)
