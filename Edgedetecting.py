
#filter sobel 
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data 

img = cv2.imread ("megachan.jpeg", 0)

#sobel
img_Sobelx = cv2.Sobel (img,cv2.CV_8U,1,0,ksize=5)
img_Sobely = cv2.Sobel(img, cv2.CV_8U,0,1,ksize=5)
img_Sobel = img_Sobelx + img_Sobely

fig, axes = plt.subplots(4, 2, figsize=(20,20))
ax = axes.ravel()

ax[0].imshow(img,cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_Sobelx,cmap ='gray')
ax[2].set_title("Citra Input")
ax[3].hist(img_Sobelx.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input")

ax[4].imshow(img_Sobely,cmap ='gray')
ax[4].set_title("Citra Input")
ax[5].hist(img_Sobely.ravel(), bins = 256)
ax[5].set_title("Histogram Citra Input")

ax[6].imshow(img_Sobel,cmap ='gray')
ax[6].set_title("Citra Input")
ax[7].hist(img_Sobel.ravel(), bins = 256)
ax[7].set_title("Histogram Citra Input")
fig.tight_layout()

#filter Prewit
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data 

img = cv2.imread ("megachan.jpeg", 0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

fig, axes = plt.subplots(4, 2, figsize=(20,20))
ax = axes.ravel()

ax[0].imshow(img,cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx,cmap ='gray')
ax[2].set_title("Citra Input")
ax[3].hist(img_prewittx.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input")

ax[4].imshow(img_prewitty,cmap ='gray')
ax[4].set_title("Citra Input")
ax[5].hist(img_prewitty.ravel(), bins = 256)
ax[5].set_title("Histogram Citra Input")

ax[6].imshow(img_prewitt,cmap ='gray')
ax[6].set_title("Citra Input")
ax[7].hist(img_prewitt.ravel(), bins = 256)
ax[7].set_title("Histogram Citra Input")
fig.tight_layout()

#fiter canny
#filter Prewit
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data 

img = cv2.imread ("megachan.jpeg", 0)

img_canny = cv2.Canny(img,100,200)

fig, axes = plt.subplots(2, 2, figsize=(20,20))
ax = axes.ravel()

ax[0].imshow(img,cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny,cmap ='gray')
ax[2].set_title("Citra Input")
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input")

fig.tight_layout()

#filter laplacian
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('megachan.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgaja = cv2.GaussianBlur(gray,(3,3),0)
laplacian = cv2.Laplacian(imgaja,cv2.CV_64F)

plt.subplot(1,2,1)
plt.imshow(imgaja, cmap='gray')
plt.title('Original')   
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])
plt.show()

img = cv2.imread('megachan.jpeg',0)
blur = cv2.GaussianBlur(img,(3,3),0)
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()
cv2.imshow('laplacian-gaussian', laplacian1)
cv2.waitKey(0)
32