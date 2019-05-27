import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#%%
rgb_dir = './train_data/eps/0/'
alpha_dir = './train_data/alpha/0/'
save_dir_sobel = './train_data/gradient_sobel/0/'
save_dir_scharr = './train_data/gradient_scharr/0/'
save_dir_laplace = './train_data/gradient_laplace/0/'

#%%
#Sobel算子
def sobel_demo(image,mask):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    gradxy[mask]=0
#    plt.figure()
#    plt.imshow(gradxy)
#    plt.show()
    return gradxy

#%%
#Scharr算子(Sobel算子的增强版，效果更突出)
def Scharr_demo(image,mask):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    gradxy[mask]=0
    return gradxy

#%%
#拉普拉斯算子
def Laplace_demo(image,mask):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    lpls[mask]=0
    return lpls

#%%
#def direction_demo(image,mask):
#    I=np.array(image)/255
#    I=np.expand_dims(I,2)
#    iter_num=3
#    w_normal=I
#    for it in range(1,iter_num+1):
#        gaborBank = gabor(5,0:5.625:180-5.625);
#        gaborMag = imgaborfilt(w_normal,gaborBank);
#        [gabor_max,gabor_id]=max(gaborMag,[],3);
#        
#        %caculate the confidence map
#        w=zeros(size(gabor_id));
#        for i=1:32
#            dis_map=dis_angle(5.625*(i-1),5.625*gabor_id);
#            w_tmp=(dis_map.*(gaborMag(:,:,i)-gabor_max).^2).^0.5;
#            w=w_tmp+w;
#        end
#        % threshold our confidence map.
#        w_max=max(max(w));
#        w_threshold=(w>w_max*0.01);
#
#        w_normal=w.*double(w_threshold);
#        
#        %normal the confidence map
#        w_normal=norm(w_normal,w_threshold);
    
    
#%%
#for fname in os.listdir(rgb_dir):
#    src = cv.imread(rgb_dir+fname,-1)
#    alpha = cv.imread(alpha_dir+fname,-1)
#    alpha = np.expand_dims(alpha,3)
#    alpha=np.concatenate([alpha,alpha,alpha],axis=2)
#    src = cv.resize(src,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
#    alpha = cv.resize(alpha,(0,0),fx=0.1,fy=0.1,interpolation=cv.INTER_NEAREST)
#    mask=alpha==0
#    gray = np.array(Image.open(rgb_dir+fname).convert('L'))
#    direction_demo(gray,mask)
#    src[mask]=0
#    gradxy1=sobel_demo(src,mask)
#    cv.imwrite(save_dir_sobel+fname, gradxy1)
#    gradxy2=Scharr_demo(src,mask)
#    cv.imwrite(save_dir_scharr+fname, gradxy2)
#    lpls=Laplace_demo(src,mask)
#    cv.imwrite(save_dir_laplace+fname, lpls)