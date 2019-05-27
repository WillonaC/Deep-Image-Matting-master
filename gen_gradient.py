import cv2 as cv
import numpy as np

#%%
#Sobel算子
def sobel_demo1(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  #x方向上的梯度
    cv.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    cv.imshow("gradient", gradxy)

src = cv.imread('./train_data/eps/0/00001.png',-1)
alpha = cv.imread('./train_data/alpha/0/00001.png',-1)
alpha = np.expand_dims(alpha,3)
alpha=np.concatenate([alpha,alpha,alpha],axis=2)
src = cv.resize(src,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
alpha = cv.resize(alpha,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
mask=alpha==0
src[mask]=0
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
sobel_demo1(src)
cv.waitKey(0)
#cv.destroyAllWindows()

#%%
#Scharr算子(Sobel算子的增强版，效果更突出)
def Scharr_demo(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x2", gradx)  #x方向上的梯度
    cv.imshow("gradient_y2", grady)  #y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient2", gradxy)
src = cv.imread('./train_data/eps/0/00001.png',-1)
alpha = cv.imread('./train_data/alpha/0/00001.png',-1)
alpha = np.expand_dims(alpha,3)
alpha=np.concatenate([alpha,alpha,alpha],axis=2)
src = cv.resize(src,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
alpha = cv.resize(alpha,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
mask=alpha==0
src[mask]=0
cv.namedWindow('input_image2', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image2', src)
Scharr_demo(src)
cv.waitKey(0)
#cv.destroyAllWindows()

#%%
#拉普拉斯算子
def Laplace_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("Laplace_demo ", lpls)
src = cv.imread('./train_data/eps/0/00001.png',-1)
alpha = cv.imread('./train_data/alpha/0/00001.png',-1)
alpha = np.expand_dims(alpha,3)
alpha=np.concatenate([alpha,alpha,alpha],axis=2)
src = cv.resize(src,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
alpha = cv.resize(alpha,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
mask=alpha==0
src[mask]=0
cv.namedWindow('input_image3', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image3', src)
Laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()