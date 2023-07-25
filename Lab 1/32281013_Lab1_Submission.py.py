#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy import nan


# In[2]:


student_id = 32281013
student_name = "Avvienash Jaganathan"


# In[3]:


"""
NOTE: None of the code were generated using an AI coding program. All code were manually written by me
"""


# # Task 1: Implement a function to perform convolution

# In[4]:


def conv(im, A):
    
    H, W = im.shape
    h, w = A.shape
    B = np.zeros((H-h+1,W-w+1))
    
    for x in range((H-h+1)*(W-w+1)):
            r = x // (W-w+1)
            c = x % (W-w+1)
            B[r,c] = np.sum(np.multiply(A,im[r:r+h,c:c+w]))
    
       
    return B


# In[5]:



"""
Assumpitions:
 - The kernel size is a multiple of the image size
 - Stride = 1
 - Padding = 0

The function takes two arguments:

    im: a 2D numpy array representing the input image.
    A: a 2D numpy array representing the filter.
    
First, the function extracts the height and width of the input image using the shape attribute of the numpy array.
It also extracts the height and width of the filter.

Then, the function creates a new numpy array B with dimensions (H-h+1, W-w+1), 
where H and W are the height and width of the input image, and h and w are the height and width of the filter. 
B is the output array that will store the result of the convolution.

Next, the function performs the convolution operation 
by looping through all the possible positions of the filter A over the input image im. 
For each position, it multiplies the filter A element-wise with the corresponding subregion of the input image im, 
and then takes the sum of the resulting values. 
This sum is then stored in the output array B at the corresponding position.

Finally, the function returns the output array B.
The output size can be determined using the formula: Output_size = Input_size - Kernel_size + 1

In summary, the conv function performs 2D convolution of an input image with a filter by sliding the filter over the image and computing the sum of element-wise products between the filter and the corresponding subregion of the image at each position.

"""


# In[6]:


kernel = (1/159) *  np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])


# In[7]:


def gaus(size,sigma):
    k = (size-1)/2
    G = np.zeros((size,size))
    for x in range(size*size):
        i = x%size;
        j = x//size;
        G[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))
        
    return G


# In[8]:


"""
Function Name: gaus

Inputs:

size: an integer representing the size of the kernel.
sigma: a float representing the standard deviation of the Gaussian distribution.

Output:

A 2D numpy array representing the Gaussian kernel.

Function Description:

Generates a 2D Gaussian kernel of the specified size and standard deviation using the Gaussian distribution formula.
Returns the generated Gaussian kernel as a 2D numpy array.

"""


# In[9]:


im = cv2.imread('./test00.png',cv2.IMREAD_GRAYSCALE)

print("The image size is: ",im.shape)
plt.imshow(im, cmap='gray')
plt.colorbar()
plt.title("Original Image")
plt.show()

im_blur = conv(im, kernel)
im_blur = (255*(im_blur - np.min(im_blur))/np.ptp(im_blur)).astype(np.uint8) 

print("The image size is: ",im_blur.shape)
plt.imshow(im_blur,cmap='gray')
plt.colorbar()
plt.title("Blured Image")
plt.show()


# # Task 2: Calculate the image gradients
# 

# In[10]:


# Write your code here
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

grad_x = conv(im_blur,sobel_x)
grad_y = conv(im_blur,sobel_y)


# In[11]:


# Show results here
print("The image size is: ",grad_x.shape)
plt.imshow(grad_x,cmap='gray')
plt.colorbar()
plt.title("Gradient X")
plt.show()

print("The image size is: ",grad_y.shape)
plt.imshow(grad_y,cmap='gray')
plt.colorbar()
plt.title("Gradient Y")
plt.show()


# In[12]:


## Demonstrate your understanding by answering the question below:
## Looking at the filter coefficients, explain how a sobel filter picks out horizontal edges?

"""
The Sobel filter for detecting horizontal edges has a row of coefficients that are designed
to highlight horizontal edges that run from right to left (-1, 0, 1), 
and a row of coefficients that are designed to highlight horizontal edges that run from left to right (1, 0, -1).

The Sobel filter detects horizontal edges by comparing the difference of pixel values between two adjacent columns in the image.

If a pixel in the image has a high value and is located to the left of a pixel with a low value, the filter will produce a large positive value in the corresponding output pixel.

If a pixel in the image has a low value and is located to the left of a pixel with a high value, the filter will produce a large negative value in the corresponding output pixel.

Overall, the Sobel filter for detecting horizontal edges works by using a difference of pixel values between two adjacent columns in the image.

The kernel coefficients are designed to highlight horizontal edges that run from right to left or from left to right, depending on the sign of the coefficients.
"""


# # Task 3: Calculate gradient magnitude

# In[13]:


grad_mag = np.hypot(grad_x,grad_y)


# In[14]:


# Show results here
print("The image size is: ",grad_mag.shape)
plt.imshow(grad_mag,cmap='gray')
plt.colorbar()
plt.title("Grad mag")
plt.show()


# In[15]:


# Demonstrate your understanding by answering the question below:
# What differences in gradient magnitude would be observed for a corner, edge and solid region?
"""
For a corner, where there is a sharp change in intensity in two perpendicular directions,
the gradient magnitude will be high in both directions,
resulting in a large gradient magnitude overall.

For an edge, where there is a gradual change in intensity in one direction,
the gradient magnitude will be high in that direction and low in the perpendicular direction,
resulting in a moderate gradient magnitude overall.

For a solid region, where there is little to no change in intensity,
the gradient magnitude will be low in all directions,
resulting in a small gradient magnitude overall.
"""


# # Task 4: Calculate gradient orientation

# In[16]:


# Write your code here
grad_ori = (np.round( np.arctan2(-grad_y,grad_x)/(np.pi/4))*45)
grad_ori[grad_ori < 0] += 180 # convert negative angles to positive
grad_ori[grad_ori == 180] = 0 # 180 and 0 are the same, so set as 0
grad_ori[np.multiply(grad_x == 0, grad_y == 0)] = 180 # if grad x and grad y are 0, The ori is undefined.


# In[17]:


# Show results here



"""
0°: | 
45°: \
90°: - 
135°: / 

"""

print("The image size is: ",grad_ori.shape)
plt.imshow(grad_ori, cmap= "ocean")
plt.colorbar()
plt.title("Gradient Orientaion rounded to nearst 45 deg")
plt.show()


# In[18]:


# Demonstrate your understanding by answering the question below:
# How could the gradient orientation be used to obtain rotational invariance for patch matching?
"""
The gradient orientation can be used to obtain rotational invariance for patch matching 
by aligning the gradients of the patches being compared.

To do this, we can calculate the gradient orientation for each patch using a Sobel filter,
as we did previously. 
Then, we can rotate each patch 
so that the dominant gradient orientation aligns with a fixed direction, 
such as the vertical axis. This is achieved by rotating the patch 
by the opposite angle of the dominant gradient orientation.

By aligning the dominant gradient orientation of each patch, 
we ensure that they have the same orientation before matching, 
and thus are more likely to be similar regardless of their orientation in the image.

Once the patches are aligned, 
we can use any patch matching algorithm, 
such as normalized cross-correlation, to compare them. 
The result of the matching algorithm will be invariant to rotations of the patches, 
as we have aligned their dominant gradient orientation.

In summary, we can use the gradient orientation to obtain rotational invariance for patch matching 
by aligning the dominant gradient orientation of each patch to a fixed direction before comparing them.

"""


# # Task 5: Extend your code to perform non-maximal suppression

# In[19]:


def non_max_sup(mag,angle):
    
    H, W = mag.shape
    M = np.zeros((H,W))
    
    for i in range((H-2)*(W-2)):
        r = i // (W-2) + 1
        c = i % (W-2) + 1
            
            
        m = mag[r,c]

        if (m == 0):
            continue

        mu = mag[r-1,c]
        mul = mag[r-1,c-1]
        mur = mag[r-1,c+1]
        mr = mag[r,c+1]
        ml = mag[r,c-1]
        md = mag[r+1,c]
        mdl = mag[r+1,c-1]
        mdr = mag[r+1,c+1]

        match angle[r,c]:
            case 0:
                if (m >= ml) and (m >= mr):
                    M[r,c] = m
            case 45:
                if (m >= mur) and (m >= mdl):
                    M[r,c] = m
            case 90:
                if (m >= mu) and (m >= md):
                    M[r,c] = m                    
            case 135:
                if (m >= mul) and (m >= mdr):
                    M[r,c] = m
    return M


# In[20]:


"""

Function Name: non_max_sup

Inputs:

mag: a 2D numpy array representing the magnitude of the edge response.
angle: a 2D numpy array representing the angle of the edge response.

Output:

A 2D numpy array representing the non-maximum suppressed edge response.

Function Description:

Performs non-maximum suppression on the edge response to thin edges to a single pixel width.
Returns the non-maximum suppressed edge response as a 2D numpy array.

"""


# In[21]:


def double_thres(M,tl,th):
    low_thres = tl
    high_thres = th 
    
    T = np.zeros_like(M)

    T[M > low_thres] = 1
    T[M > high_thres] = -1
    
    return T


# In[22]:


"""

Function Name: double_thres

Inputs:

M: a 2D numpy array representing the non-maximum suppressed edge response.
tl: a float representing the low threshold value.
th: a float representing the high threshold value.

Output:

A 2D numpy array representing the thresholded edge response.

Function Description:

Applies double thresholding to the non-maximum suppressed edge response to identify strong and weak edges.
Returns a binary image where the strong edges are set to -1 and the weak edges are set to 1.

"""


# In[23]:


def single_thres(M,t = 0.2):
    
    thres = t * np.max(M)
    
    T = np.zeros_like(M)

    T[M > thres] = 1
   
    
    return T


# In[24]:


"""

Function Name: single_thres

Inputs:

M: a 2D numpy array representing the non-maximum suppressed edge response.
t: a float representing the threshold value. (Optional, default value is 0.2)

Output:

A 2D numpy array representing the thresholded edge response.

Function Description:

Applies single thresholding to the non-maximum suppressed edge response to identify edges.
Returns a binary image where the edges are set to 1 and the non-edges are set to 0.

"""


# In[25]:


def connectivity_analysis_4_connected(M):
    
    labels = {}
    
    
    l = 1
    rm  = np.array([])
    T = np.zeros_like(M)
    h,w = M.shape
    
    for r in range(h):
        for c in range(w):
            
            m = M[r,c]
            if m == 0:
                continue
            
            if (r == 0) and (c  == 0):
                t_n = np.array([T[r,c+1],T[r+1,c]])
            elif (r == h-1) and (c  == 0):
                t_n = np.array([T[r-1,c],T[r,c+1]])
            elif (r == 0) and (c  == w-1):
                t_n = np.array([T[r,c-1],T[r+1,c]])
            elif (r == h-1) and (c  == w-1):
                t_n = np.array([T[r-1,c],T[r,c-1]])
            elif (r == 0):
                t_n = np.array([T[r,c+1],T[r,c-1],T[r+1,c]])
            elif (c  == 0):
                t_n = np.array([T[r-1,c],T[r,c+1],T[r+1,c]])
            elif (r == h-1):
                t_n = np.array([T[r-1,c],T[r,c+1],T[r,c-1]])
            elif (c  == w-1):
                t_n = np.array([T[r-1,c],T[r,c-1],T[r+1,c]])
            else:
                t_n = np.array([T[r-1,c],T[r,c+1],T[r,c-1],T[r+1,c]])
         
            
            for n in range(np.prod(t_n.shape)):
                if (t_n[n] != 0) and (T[r,c] == 0):
                    T[r,c] = t_n[n]
                elif (t_n[n] != 0) and (T[r,c] != 0) and (T[r,c] != t_n[n]):
                    T_max = np.max([T[r,c],t_n[n]])
                    T[r,c] = np.min([T[r,c],t_n[n]])
                    labels[T[r,c]] = np.unique(np.concatenate((labels[T[r,c]], labels[T_max]))).astype(int)
                    rm = np.unique(np.append(rm, T_max))


            
            if T[r,c] == 0:
                T[r,c] = l
                labels[l] = np.array([l])
                l += 1
    
                                                                                     
        

    for lb in rm:
        labels.pop(lb)

                        
    for r in range(h):
        for c in range(w):
            t = T[r,c]
            if t == 0:
                continue
            for key in labels.keys():
                arr = labels[key]
                for lb in arr:
                    if t == lb:
                        t = key
                        break
            T[r,c] = t
    
    M_c = np.zeros_like(M)
    M_c[M == -1] = 1
    for lb in labels.keys():
        if (np.sum(np.multiply((M == -1),(T == lb))) > 0):
            M_c[np.multiply((M == 1),(T == lb))] = 1
    
    return M_c,T


# In[26]:


"""
Function Name: connectivity_analysis_4_connected

Inputs:

M: a 2D numpy array representing a binary image, where 0 represents the background and 1 represents the foreground.

Outputs:

labels: a dictionary where the keys are the integer labels for each connected component in the image and the values are numpy arrays containing the coordinates of each pixel belonging to the component.

Function Description:

This function performs 4-connected connectivity analysis on a binary image and returns the labels for each connected component.
Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

"""


# In[27]:


def connectivity_analysis_8_connected(M):
    
    labels = {}
    
    
    l = 1
    rm  = np.array([])
    T = np.zeros_like(M)
    h,w = M.shape
    
    for r in range(h):
        for c in range(w):
            
            m = M[r,c]
            if m == 0:
                continue
            
            if (r == 0) and (c  == 0):
                t_n = np.array([T[r,c+1],T[r+1,c],T[r+1,c+1]])
            elif (r == h-1) and (c  == 0):
                t_n = np.array([T[r-1,c],T[r-1,c+1],T[r,c+1]])
            elif (r == 0) and (c  == w-1):
                t_n = np.array([T[r,c-1],T[r+1,c],T[r+1,c-1]])
            elif (r == h-1) and (c  == w-1):
                t_n = np.array([T[r-1,c],T[r-1,c-1],T[r,c-1]])
            elif (r == 0):
                t_n = np.array([T[r,c+1],T[r,c-1],T[r+1,c],T[r+1,c-1],T[r+1,c+1]])
            elif (c  == 0):
                t_n = np.array([T[r-1,c],T[r-1,c+1],T[r,c+1],T[r+1,c],T[r+1,c+1]])
            elif (r == h-1):
                t_n = np.array([T[r-1,c],T[r-1,c-1],T[r-1,c+1],T[r,c+1],T[r,c-1]])
            elif (c  == w-1):
                t_n = np.array([T[r-1,c],T[r-1,c-1],T[r,c-1],T[r+1,c],T[r+1,c-1]])
            else:
                t_n = np.array([T[r-1,c],T[r-1,c-1],T[r-1,c+1],T[r,c+1],T[r,c-1],T[r+1,c],T[r+1,c-1],T[r+1,c+1]])
                
         
            
            for n in range(np.prod(t_n.shape)):
                if (t_n[n] != 0) and (T[r,c] == 0):
                    T[r,c] = t_n[n]
                elif (t_n[n] != 0) and (T[r,c] != 0) and (T[r,c] != t_n[n]):
                    T_max = np.max([T[r,c],t_n[n]])
                    T[r,c] = np.min([T[r,c],t_n[n]])
                    labels[T[r,c]] = np.unique(np.concatenate((labels[T[r,c]], labels[T_max]))).astype(int)
                    rm = np.unique(np.append(rm, T_max))


            
            if T[r,c] == 0:
                T[r,c] = l
                labels[l] = np.array([l])
                l += 1
    
                                                                                     
        

    for lb in rm:
        labels.pop(lb)

                        
    for r in range(h):
        for c in range(w):
            t = T[r,c]
            if t == 0:
                continue
            for key in labels.keys():
                arr = labels[key]
                for lb in arr:
                    if t == lb:
                        t = key
                        break
            T[r,c] = t
    
    M_c = np.zeros_like(M)
    M_c[M == -1] = 1
    for lb in labels.keys():
        if (np.sum(np.multiply((M == -1),(T == lb))) > 0):
            M_c[np.multiply((M == 1),(T == lb))] = 1
    
    return M_c,T


# In[28]:


"""
Function Name: connectivity_analysis_8_connected

Inputs:

M: a 2D numpy array representing a binary image, where 0 represents the background and 1 represents the foreground.

Outputs:

labels: a dictionary where the keys are the integer labels for each connected component in the image and the values are numpy arrays containing the coordinates of each pixel belonging to the component.

Function Description:

This function performs 8-connected connectivity analysis on a binary image and returns the labels for each connected component.
Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

"""


# In[29]:


def median_filter(img_noisy):

    m, n = img_noisy.shape

    # Traverse the image. For every 3X3 area, 
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img_noisy[i-1, j-1],
                   img_noisy[i-1, j],
                   img_noisy[i-1, j + 1],
                   img_noisy[i, j-1],
                   img_noisy[i, j],
                   img_noisy[i, j + 1],
                   img_noisy[i + 1, j-1],
                   img_noisy[i + 1, j],
                   img_noisy[i + 1, j + 1]]

            temp = sorted(temp)
            img_new[i, j]= temp[4]
    return img_new


# In[30]:


"""

Name: median_filter

Inputs:

img_noisy: a 2D numpy array representing the noisy image.

Outputs:

img_new: a 2D numpy array representing the filtered image.

Description:

The median_filter function applies a median filter to a noisy image.
It then computes the median value of the pixel values in the window and replaces the center pixel value with the median value.
Good for salt and pepper noise

"""


# In[31]:


def adaptive_filtering(M,kernel_size,mu):

       
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    local_mean = conv(M, kernel)

    local_var = conv(np.square(M),kernel) - np.square(local_mean)

    scaling_factor = mu / (local_var + mu)
    
    h, w = M.shape
    n = (kernel_size-1)//2

    M_new = local_mean + scaling_factor * (M[n:h-n,n:w-n] - local_mean)

    return M_new


# In[32]:


"""

Name: adaptive_filtering

Inputs:

M: a 2D numpy array representing the input image.
kernel_size: an integer representing the size of the kernel (must be odd).
mu: a float representing a constant that is used in the scaling factor.

Outputs:

M_new: a 2D numpy array representing the filtered image.

Description:
The adaptive_filtering function applies an adaptive filter to an input image M.
"""


# In[ ]:


M_non_max = non_max_sup(grad_mag,grad_ori)

print("The image size is: ",M_non_max.shape)
plt.imshow(M_non_max,cmap='gray') #tab20c
plt.colorbar()
plt.title("Non Max Sup")
plt.show()


th = np.sum(M_non_max)/np.sum([M_non_max != 0])
tl = np.mean(M_non_max)
M_thres = double_thres(M_non_max,tl,th)

plt.imshow(np.array(M_thres),cmap='gray') #tab20c
plt.colorbar()
plt.title("Threshold")
plt.show()

M_connected,T = connectivity_analysis_4_connected(M_thres)

plt.imshow(T,cmap='tab20c') #tab20c
plt.colorbar()
plt.title("Connectivity Blobs")
plt.show()


plt.imshow(M_connected,cmap='gray') #tab20c
plt.colorbar()
plt.title("Final")
plt.show()


# In[ ]:


# Demonstrate your understanding by answering the question below:
# Explain how you chose the threshold for non-maximal suppression?
"""
For the upper threshold, I used the mean of all the non zero grad magnitude values multipled by a factor
For the lower threshold, I used the mean of all the grad magnitude values multipled by a factor

"""


# # Task 6: Find the helipad

# In[ ]:


# Write your code here
im = cv2.imread('./task6_helipad.png',cv2.IMREAD_GRAYSCALE)

print("The image size is: ",im.shape)

plt.imshow(im, cmap='gray')
plt.colorbar()
plt.title("Original Image")
plt.show()

im_denoised =  median_filter(im)

print("The image size is: ",im_denoised.shape)
print("Type: ",im_denoised.dtype)

plt.imshow(im_denoised, cmap='gray')
plt.colorbar()
plt.title("Denoised Image")
plt.show()


#im_blur = adaptive_filtering(im_denoised,11,100)
im_blur = conv(im_denoised,gaus(20,3))


print("The image size is: ",im_blur.shape)

plt.imshow(im_blur,cmap='gray')
plt.colorbar()
plt.title("Blured Image")
plt.show()

grad_x = conv(im_blur,sobel_x)
grad_y = conv(im_blur,sobel_y)

print("The image size is: ",grad_x.shape)

plt.imshow(grad_x,cmap='gray')
plt.colorbar()
plt.title("Grad X")
plt.show()

print("The image size is: ",grad_y.shape)

plt.imshow(grad_y,cmap='gray')
plt.colorbar()
plt.title("Grad Y")
plt.show()

grad_mag = np.hypot(grad_x,grad_y)

print("The image size is: ",grad_mag.shape)

plt.imshow(grad_mag,cmap='gray')
plt.colorbar()
plt.title("Grad mag")
plt.show()


grad_ori = (np.round( np.arctan2(-grad_y,grad_x)/(np.pi/4))*45)
grad_ori[grad_ori < 0] += 180 # convert negative angles to positive
grad_ori[grad_ori == 180] = 0 # 180 and 0 are the same, so set as 0
grad_ori[np.multiply(grad_x == 0, grad_y == 0)] = 180 # if grad x and grad y are 0, The ori is undefined.

print("The image size is: ",grad_ori.shape)

# plt.imshow(np.array(grad_ori == 135)*1)
plt.imshow(grad_ori, cmap= "ocean")
plt.colorbar()
plt.title("Grad Ori")
plt.show()


M_nm = non_max_sup(grad_mag,grad_ori)
print("The image size is: ",M_nm.shape)

plt.imshow(M_nm,cmap='gray') #tab20c
plt.colorbar()
plt.title("Non Max Sup")
plt.show()


th = 1.5*np.sum(M_nm)/np.sum([M_nm != 0])
tl = np.mean(M_nm)
M_t = double_thres(M_nm,tl,th)
print([th,tl])

print("The image size is: ",M_t.shape)
plt.imshow(np.array(M_t),cmap='gray') #tab20c
plt.colorbar()
plt.title("Doube Threshold")
plt.show()

M_c,T = connectivity_analysis_4_connected(M_t)

print("The image size is: ",M_c.shape)

plt.imshow(M_c,cmap='gray') #tab20c
plt.colorbar()
plt.title("Final")
plt.show()

moment_x = 0
moment_y = 0
count = np.sum(M_c)

for r in range(M_c.shape[0]):
    for c in range(M_c.shape[1]):
        moment_x += c*M_c[r,c]
        moment_y += r*M_c[r,c]

c_x = moment_x /count
c_y = moment_y /count

plt.plot(c_x, c_y, marker="x", color="yellow",markersize=15)
plt.imshow(M_c,cmap='gray')
plt.show()

H, W = im.shape
h, w = M_c.shape

c_x +=(W-w)/2
c_y += (H-h)/2

plt.plot(c_x, c_y, marker="x", color="blue",markersize=15)
plt.imshow(im,cmap='gray')
plt.show()

text = 'Center X:' + str(c_x) + "   Center Y : " + str(c_y) 
print(text)


# In[ ]:


# Demonstrate your understanding by answering the question below:
# How did you adapt your previous code to solve this problem?  

"""

First, I applied a median filter to remove the salt and pepper noise present in the image.

Next, I tried using a Gaussian function and adaptive filtering for better denoising, but found that the gaus and median filter was sufficient in this case.

To perform edge detection, I used the Canny edge detection algorithm which involves double thresholding. I experimented with different threshold values.

To connect the edges and remove any small gaps, I performed 8-connectivity analysis. However, I found that using 4-connectivity resulted in a smoother and more accurate edge detection result.

To track the edges, I used the hysteresis technique which involves selecting strong edges above the upper threshold and connecting them to weak edges above the lower threshold.

Finally, after processing the image, I noticed that the center of the resulting shape was slightly lower than expected. I realized that this was due to the length of the bottom legs being slightly longer than the top legs after the image processing.

"""


# # Do not remove or edit the following code snippet. 
# 
# When submitting your lab, please ensure that you have run the entire notebook from top to bottom. You can do this by clicking "Kernel" and "Restart Kernel and Run All Cells". Make sure the last cell (below) has also been run. 

# In[ ]:


file_name = str(student_id) + '_Lab1_Submission.py'
cmd = "jupyter nbconvert --to script Lab1_student_template.ipynb --output " + file_name
if(os.system(cmd)):
    print("Error converting to .py")
    print("cmd")

