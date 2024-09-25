#python iterators are objects that can be iterated upon
#to convert a datastructure into an iterator you can use the iter() function
#example

tuple1 = (1,2,3,4,5)
myiter = iter(tuple1)
print(next(myiter))

#__iter__() is used to initialize the iterator
#__next__() is used to get the next item in the sequence

#in machine learning we use iterators to iterate over the dataset
#in pytorch we use the dataloader class to create an iterator
#example

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

data = datasets.MNIST(root = 'data',train = True,download = True)
data_loader = DataLoader(data,batch_size = 32,shuffle = True)

