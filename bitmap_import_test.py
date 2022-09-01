# Python program to read
# image using matplotlib
import numpy as np
from PIL import Image

# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import PIL


# Read Images
img2 = mpimg.imread('MD_resize.png')
img = mpimg.imread('test_image.png')

print(img.shape)
rows = 2
columns = 2

fig = plt.figure()

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(img)

#new_h, new_w = int(img.shape[0] / 2), int(img.shape[1] / 2)
#img = img.shape[new_h, new_w,:]
#print("new image", img)



# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(img2)

#fig.show()
#plt.show()

#print arrays of color
#print(img)

def color_index_x_y(img):
    for color in img:
        print("color ", color)
    for x, y in range(len(img)):
        x = img[i]
        print("xcords,", x)
    #for px in img[i]:

X = np.zeros([1,3])
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        rgb = np.array(img[y,x,:3])
        if not((rgb[0] == 0) and (rgb[1] == 0) and (rgb[2] == 0)):
            if((rgb[0]==0) and (rgb[1]==0) and (rgb[2]==1)):
                X = np.vstack([X,[x,y,1]]) # blue is 1
            elif ((rgb[0] == 1) and (rgb[1] == 0) and (rgb[2] == 0)):
                X = np.vstack([X,[x,y,2]]) # red is 2
            elif ((rgb[0] == 0) and (rgb[1] == 1) and (rgb[2] == 0)):
                X = np.vstack([X,[x, y, 3]]) # green is 3
            else:
                X = np.vstack([X,[x, y, 0]]) # eveything else (including white) is 0

np.savez('img1.npz',X=X)

'''
X = np.zeros([3,1])
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        rgb = np.array(img[y,x,:3])
        if((rgb[0]!=0) and (rgb[1]!=0) and (rgb[2]!=0)):
            #print(rgb)
            if((rgb[0]==0) and (rgb[2]==1)):
                print('blue!')


print(img)
sds

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if (img[y, x, :3] != [0,0,0]):
            print(img[y, x, :3])




#color_index_x_y(img)
#print("location test", img[100][100])
'''