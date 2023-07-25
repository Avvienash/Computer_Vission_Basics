#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[2]:


student_id = 32281013
student_name = "Avvienash Jaganathan"


# # Task 1: Draw test points on the left image

# In[3]:


# Write your code here
im_l = cv2.cvtColor(cv2.imread('left.jpg'), cv2.COLOR_BGR2GRAY)
im_r = cv2.cvtColor(cv2.imread('right.jpg'), cv2.COLOR_BGR2GRAY)

plt.imshow(im_l ,cmap='gray')
plt.title("Left Image")
plt.show()

plt.imshow(im_r ,cmap='gray')
plt.title("Right Image")
plt.show()


# In[4]:


test_points_left = np.array([[337,196,1],
                            [467,289,1],
                            [252,169,1],
                            [262,255,1],
                            [241,135,1]])
test_points_left = np.transpose(test_points_left)

print("Test Points:")
print(test_points_left)


# In[5]:


plt.imshow(im_l,cmap='gray')
for i in range(test_points_left.shape[1]):
    x = test_points_left[0,i]/test_points_left[2,i]
    y = test_points_left[1,i]/test_points_left[2,i]
    plt.plot(x, y, marker="x", color="red",markersize=15)

plt.title("Test Points Left")
plt.show()


# # Task 2: Use Homography to find right image points
# 

# In[6]:


H = np.array([[1.6010,-0.0300,-317.9341],
             [0.1279,1.5325,-22.5847],
             [0.0007,0,1.2865]])
print("H:")
print(H)


# In[7]:


test_points_right = np.matmul(H,test_points_left)
print("Test Points Right:")
print(test_points_right)


# In[8]:


plt.imshow(im_r,cmap='gray')
for i in range(test_points_right.shape[1]):
    x = test_points_right[0,i]/test_points_right[2,i]
    y = test_points_right[1,i]/test_points_right[2,i]
    plt.plot(x, y, marker="x", color="red",markersize=15)

plt.title("Test Points Right")
plt.show()


# In[9]:


# Demonstrate your understanding by answering the question below:
# Describe the process of calculating a homography matrix. Ensure you list out the key steps

"""
1) Obtain at least 4 matches
2) Build a matrix from the u,v x,y values
3) Calculate the null-space of hij values
4) Repack the values into a matrix to get H

"""


# # Task 3: Bilinear interpolation of the right image

# In[10]:


points_task3 = np.array([test_points_right[0,:]/test_points_right[2,:],test_points_right[1,:]/test_points_right[2,:]])
print(points_task3)


# In[11]:


def bilinear_interpolations(image,points):
    
    x = points[0,:]
    y = points[1,:]
    
    x1 = np.floor(points[0,:]).astype(int)
    y1 = np.floor(points[1,:]).astype(int)
    x2 = np.ceil(points[0,:]).astype(int)
    y2 = np.ceil(points[1,:]).astype(int)
    
    w1 = (x2-x) * (y2-y)
    w2 = (x-x1) * (y-y1)
    w3 = (x-x1) * (y2-y)
    w4 = (x2-x) * (y-y1)
        
    v1 = image[y1,x1]*w1
    v2 = image[y2,x2]*w2
    v3 = image[y1,x2]*w3
    v4 = image[y2,x1]*w4
    
    
    
    return v1+v2+v3+v4

def bilinear_interpolation(image,point):
    
    x = point[0]
    y = point[1]
    
    x1 = np.floor(point[0]).astype(int)
    y1 = np.floor(point[1]).astype(int)
    x2 = np.ceil(point[0]).astype(int)
    y2 = np.ceil(point[1]).astype(int)
    
    w1 = (x2-x) * (y2-y)
    w2 = (x-x1) * (y-y1)
    w3 = (x-x1) * (y2-y)
    w4 = (x2-x) * (y-y1)
    
    v1 = image[y1,x1]
    v2 = image[y2,x2]
    v3 = image[y1,x2]
    v4 = image[y2,x1]
    
    return np.sum([v1*w1,v2*w2,v3*w3,v4*w4])
    


# In[12]:


interpolated_intensity_values = bilinear_interpolations(im_r,points_task3)

print(interpolated_intensity_values)


# In[13]:


# Demonstrate your understanding by answering the question below:
# Why is bilinear interpolation necessary? 
"""
After performing the Homography to find the points in right image. the points are non integrers
Hence, it does not correspond to a particular pixel so we need to use Bilinear intyerpolation to get the value at the point 
using the values of its neighbours
"""
#Which equation did you use to calculate the bilinear interpolation 

"""
 = image[y1,x1]*w1 + image[y2,x2]*w2 + image[y2,x1]*w3 + image[y1,x2]*w4

where the coordinates are the nearest pixels values (top left, top right,bottom left, bottom right) 
The weights are based on the distance between the target point and the four nearest pixels.
"""
#and why?

"""
 It is assumed that the intensity values vary linearly in both the x and y directions between pixels.
"""


# # Task 4: Image stitching

# In[14]:


width = 1024
height = 384

im_new = np.ones((height,width))*-1
im_new[:,:im_l.shape[1]] = im_l[:,:]

for r in range(height):
    for c in range(im_l.shape[1],width ):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            im_new[r,c] = bilinear_interpolation(im_r,[x,y])

plt.figure(figsize=(10,10))
plt.imshow(im_new ,cmap='gray')
plt.title("New Image")
plt.show()


# In[14]:


# Demonstrate your understanding by answering the question below:
# Why are some pixels invalid after applying the homography matrix
"""
After Transformation it is possible the point falls outside the image baoundaries
"""


# # Task 5: Better blending

# In[15]:


width = 1024
height = 384

im_new = np.ones((height,width))*-1
im_new[:,:im_l.shape[1]] = im_l[:,:]

for r in range(height):
    for c in range(im_l.shape[1],width ):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            im_new[r,c] = bilinear_interpolation(im_r,[x,y])
            
            

for c in range(1023,-1,-1):
    if np.sum(im_new[:,c] == -1) > 0:
        width = c
        continue
    break
    
im_new = im_new[:,:width]
print(np.max(im_new))

plt.figure(figsize=(10,10))
plt.imshow(im_new ,cmap='gray')
plt.title("New Image")
plt.show()


# In[16]:


l_i = 1
r_i = 1



while True:
    seam_l = np.copy(im_new[:,im_l.shape[1]-1])*l_i # get values to the left of the seam
    seam_r = np.copy(im_new[:,im_l.shape[1]])*r_i # get values to the right of the seam
    
    diff = np.mean(seam_l - seam_r)
    
    scale = 10000
    l_i -= diff/scale
    r_i += diff/scale

    if abs(diff) < 0.0000000001:
        break
        
    if abs(diff) > 200:
        print("Unstable")
        break
        
print("diff:",diff)
print("Left Image Brightnest Scale:",l_i)
print("Right Image Brightnest Scale:",r_i)
    


im_new_brightness = np.copy(im_new)
im_new_brightness[:,:im_l.shape[1]] *= l_i
im_new_brightness[:,im_l.shape[1]:] *= r_i


plt.figure(figsize=(10,10))
plt.imshow(im_new_brightness ,cmap='gray')
plt.title("New Image Brightness Adjusted")
plt.show()


# In[17]:


min_c = width

for c in range(width):
    for r in range(height):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            min_c = np.min(np.array([min_c,c]))

            
max_c = im_l.shape[1]

overlap_width = max_c - min_c


im_new_blend = np.copy(im_new_brightness)

for r in range(height):
    for c in range(min_c,max_c):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        
        alpha = (max_c - c)/overlap_width
        
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            im_new_blend[r,c] = im_l[r,c]*l_i*alpha + bilinear_interpolation(im_r,[x,y])*r_i*(1-alpha)

            
plt.figure(figsize=(10,10))
plt.imshow(im_new_blend ,cmap='gray')
plt.title("New Image Blended")
plt.show()


# In[19]:


"""
Not In Use, Since I decided to blend the images instead of blur the seam. But ill keep the functions here just in case
"""

def conv(im, A):
    
    H, W = im.shape
    h, w = A.shape
    B = np.zeros((H-h+1,W-w+1))
    
    for x in range((H-h+1)*(W-w+1)):
            r = x // (W-w+1)
            c = x % (W-w+1)
            B[r,c] = np.sum(np.multiply(A,im[r:r+h,c:c+w]))
    
       
    return B

def gaus(size,sigma):
    k = (size-1)/2
    G = np.zeros((size,size))
    for x in range(size*size):
        i = x%size;
        j = x//size;
        G[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))
        
    return G


# In[20]:


# Demonstrate your understanding by answering the question below:
# Describe the steps you have used to improve the blending process. Why were they effective? 

"""
Step 1: 
Remove Black edges. 
This helps ensure the stiched image is in frame by cropping it

Step 2: 
Intensity Matching. I minimized the diffrents in intensity at th seam and applied the scale to the image.
This helped ,match the brightness and contrast of the images

Step 3:
Blending
I blended the the two images together at the overlap region to remove the seam.

"""


# # Task 6: Now try your own!

# In[18]:


im_l = cv2.cvtColor(cv2.imread('task6_l.jpg'),cv2.COLOR_BGR2GRAY)
im_r = cv2.cvtColor(cv2.imread('task6_r.jpg'), cv2.COLOR_BGR2GRAY)

scale_percent = 20 # percent of original size
width = int(im_l.shape[1] * scale_percent / 100)
height = int(im_l.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
im_l = cv2.resize(im_l, dim, interpolation = cv2.INTER_AREA)
im_r = cv2.resize(im_r, dim, interpolation = cv2.INTER_AREA)
   

plt.figure(figsize=(10,10))
plt.imshow(im_l ,cmap='gray')
plt.title("Left Image")
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(im_r ,cmap='gray')
plt.title("Right Image")
plt.show()
    
orb = cv2.ORB_create(10000)
sift = cv2.SIFT_create() 
# surf = cv2.xfeatures2d.SURF_create()
   

kpl, desl = orb.detectAndCompute(im_l,None)
kpr, desr = orb.detectAndCompute(im_r,None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desl, desr)
matches = sorted(matches, key = lambda x:x.distance)


# In[19]:


src_points = []
dst_points = []

for m in matches[:500]:
    
    iml_idx = m.queryIdx
    imr_idx = m.trainIdx

    (xl, yl) = kpl[iml_idx].pt
    (xr, yr) = kpr[imr_idx].pt

    # Append to each list
    src_points.append((xl, yl))
    dst_points.append((xr, yr))

matched_im  = np.zeros((im_l.shape[0],im_l.shape[1]*2))
matched_im[:,:im_l.shape[1]] = im_l
matched_im[:,im_l.shape[1]:] = im_r

plt.figure(figsize=(50,50))
plt.imshow(matched_im ,cmap='gray')
plt.title("New Image Macthed")



for i in range(len(src_points)):
    plt.plot([src_points[i][0],dst_points[i][0]+im_l.shape[1]], [src_points[i][1], dst_points[i][1]], marker="o", color="yellow",markersize=10)
    
plt.show()


# In[20]:


src_points = np.array(src_points)
dst_points = np.array(dst_points)

H, status = cv2.findHomography(src_points,dst_points)

H


# In[21]:


test_points_left = np.vstack((np.transpose(src_points),np.ones((1,src_points.shape[0]))))

test_points_right = np.matmul(H,test_points_left)
print("Test Points Right:")
plt.imshow(im_r,cmap='gray')
for i in range(test_points_right.shape[1]):
    x = test_points_right[0,i]/test_points_right[2,i]
    y = test_points_right[1,i]/test_points_right[2,i]
    plt.plot(x, y, marker="o", color="yellow",markersize=1)

plt.title("Test Points Right")
plt.show()


# In[22]:


width = im_l.shape[1]*2
height = im_l.shape[0]

im_new = np.ones((height,width))*-1
im_new[:,:im_l.shape[1]] = im_l[:,:]

for r in range(height):
    for c in range(im_l.shape[1],width):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            im_new[r,c] = bilinear_interpolation(im_r,[x,y])
            
            

for c in range(im_l.shape[1]*2 - 1,-1,-1):
    if np.sum(im_new[:,c] == -1) == im_l.shape[0] :
        width = c
        continue
    break
    
im_new = im_new[:,:width]
im_new = np.clip(im_new,0,255)

plt.figure(figsize=(20,20))
plt.imshow(im_new ,cmap='gray')
plt.title("New Image")
plt.show()


# In[23]:


l_i = 1
r_i = 1



while True:
    seam_l = np.copy(im_new[:,im_l.shape[1]-1])*l_i # get values to the left of the seam
    seam_r = np.copy(im_new[:,im_l.shape[1]])*r_i # get values to the right of the seam
    
    diff = np.mean(seam_l - seam_r)
    
    scale = 10000
    l_i -= diff/scale
    r_i += diff/scale

    if abs(diff) < 0.0000000001:
        break
        
    if abs(diff) > 200:
        print("Unstable")
        break
        
print("diff:",diff)
print("Left Image Brightnest Scale:",l_i)
print("Right Image Brightnest Scale:",r_i)
    


im_new_brightness = np.copy(im_new)
im_new_brightness[:,:im_l.shape[1]] *= l_i
im_new_brightness[:,im_l.shape[1]:] *= r_i


plt.figure(figsize=(10,10))
plt.imshow(im_new_brightness ,cmap='gray')
plt.title("New Image Brightness Adjusted")
plt.show()


# In[24]:


min_c = width

for c in range(width):
    for r in range(height):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            min_c = np.min(np.array([min_c,c]))

            
max_c = im_l.shape[1]

overlap_width = max_c - min_c


im_new_blend = np.copy(im_new_brightness)

for r in range(height):
    for c in range(min_c,max_c):
        left_point = np.array([[c],[r],[1]])
        right_point = np.matmul(H,left_point)
        x = right_point[0]/right_point[2]
        y = right_point[1]/right_point[2]
        
        alpha = (max_c - c)/overlap_width
        
        if (np.ceil(x)<im_r.shape[1]) and (np.floor(x)>=0) and (np.ceil(y)<im_r.shape[0]) and (np.floor(x)>=0):
            im_new_blend[r,c] = im_l[r,c]*l_i*alpha + bilinear_interpolation(im_r,[x,y])*r_i*(1-alpha)

            
plt.figure(figsize=(20,20))
plt.imshow(im_new_blend ,cmap='gray')
plt.title("New Image Blended")
plt.show()


# In[ ]:


# Demonstrate your understanding by answering the question below:
# How did you adapt your previous code to solve this problem? 
"""
Step 1: detect Features
Step 2: Match Features
Step 3: Create Homography Mat
Step 4: Stich
Step 5: Crop
Step 6: Adjust Brigghnest
Step 7: Alpha Blend
"""


# # Do not remove or edit the following code snippet. 
# 
# When submitting your lab, please ensure that you have run the entire notebook from top to bottom. You can do this by clicking "Kernel" and "Restart Kernel and Run All Cells". Make sure the last cell (below) has also been run. However, you may need to download the required python libraries to do this first

# If the code snippet does not run and you do not know how to download the required python libraries, you can do this instead:
# 
# 1. Upload your completed jupyter notebook to Google Colab
# 2. Download your notebook as a .py file

# In[ ]:


file_name = str(student_id) + '_Lab2_Submission'
cmd = "jupyter nbconvert --to script Lab2_student_template.ipynb --output " + file_name
if(os.system(cmd)):
    print("Error converting to .py")
    print("cmd")

