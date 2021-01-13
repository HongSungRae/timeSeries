import torch
import numpy as np

def divide_ten(x):
    return x/10

def minmax(x):
    return (x-min(x))/(max(x)-min(x))