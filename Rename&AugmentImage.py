
# coding: utf-8

# In[1]:


import Augmentor
import os
import re
import glob
import numpy as np


# In[2]:


def renaming():
    i = 0
    label = []
    for filename in glob.glob('output/*.png'):
        i += 1
        Reg = r"output\\ss_original_O(\d+)"
        m = re.search(Reg, filename)
        temp = int(m.group(1))
        if temp < 16 or (temp < 45 and temp > 30) :
            newName = 'O' + str(i) + ".png"
            os.rename(filename, newName)
            label.append(0)
        else:
            newName = 'O' + str(i) + ".png"
            os.rename(filename, newName)
            label.append(1)
    np.savetxt('labels.txt', np.array(label)[None], fmt="%d", delimiter=",") 


# In[3]:


path = r"C:\Users\e6ncbcy\Desktop\ss"
p = Augmentor.Pipeline(path)


# In[4]:


p.shear(probability=0.4, max_shear_left=0.6, max_shear_right=0.6)
p.skew(probability=0.4, magnitude=0.02)
p.random_distortion(probability=0.4, grid_width=10, grid_height=10, magnitude=2)
# p.gaussian_distortion(probability=0.3, grid_width=1, grid_height=1, magnitude=2, corner="bell", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
p.rotate(probability=0.4, max_left_rotation=2, max_right_rotation=2)


# In[5]:


p.sample(2000)


# In[7]:


renaming()

