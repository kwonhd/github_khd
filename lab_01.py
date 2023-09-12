# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 00:16:12 2023

@author: gureh
"""

# %%
import tensorflow as tf
# %%
print(tf.__version__)
#%%
hello = tf.constant("Hello,TensorFlow!")
#%%
print(hello.numpy())
#%%
node1 =tf.constant(3.0,tf.float32)
node2 =tf.constant(4.0)
node3 =tf.add(node1,node2)

print('node1:',node1,"node2:",node2,"node3:",node3)
#%% # session 사라짐 numpy로 대체
print('node1:', node1.numpy(), "node2:", node2.numpy())
print('node3:', node3.numpy())
#%%
def adder(a,b):
    return a+b

a = tf.constant([1,3])
b = tf.constant([2,4])

print(adder(a,b))
