import numpy as np
import torch as tr
from torch.utils.data import Dataset
import torch.nn as nn

def Convolve(img: np.ndarray,kernel:np.ndarray,pad:bool=True,stride:tuple=(1,1))->np.ndarray:
    '''
    Convolve function performs the convolution operation on the input image with the given kernel.
    
    Args:
        img (np.ndarray):        The input image.
        kernel (np.ndarray):     The convolution kernel.
        pad (bool, optional):    Whether to pad the image or not. Default is True.
                                 Image is padded with zeros by default
        stride (tuple,optional): It is the step size by which the kernel is slided over 
                                 the image during convolution. The default value is set to
                                 (1,1) pixel units.It is a whole number and helps to downsample the image or decrease its size.

    Returns:
        np.ndarray: The convolved image.
    '''
    kernel_size=kernel.shape[0] ### get the size of kernel (usually square kernel with odd size).
    padding=kernel_size//2 ### decide the padding size.

    if pad:
        ### creating an ndarray of zeros of shape of image with padding appended on each side.
        if len(img.shape)==3:
            padded_img=np.zeros((img.shape[0]+2*padding,img.shape[1]+2*padding,3))
        else:
            padded_img=np.zeros((img.shape[0]+2*padding,img.shape[1]+2*padding))
        padded_img[padding:padded_img.shape[0]-padding,padding:padded_img.shape[1]-padding]=img
    else:
        ### if there is padding, then padding image is the original image.
        padded_img=img

    ### for both 2D and 3D convolution, the result is 2D.
    ### considering effect of padding and stride
    Nh= int(np.floor((img.shape[0]+2*padding-kernel_size)/stride[0]))+1
    Nw= int(np.floor((img.shape[1]+2*padding-kernel_size)/stride[1]))+1

    result=np.zeros(shape=(Nh,Nw))
    ### for 3D convolution, create a 3D kernel for given 1D to apply convolution to all three channels.
    kernel=np.dstack((kernel,kernel,kernel)) if len(padded_img.shape)==3 else kernel

    ### leaving sufficient place (padding) perform convolution (dot product) and store the value in result.
    for i in range(padding,padded_img.shape[0]-padding,stride[0]):
        for j in range(padding,padded_img.shape[1]-padding,stride[1]):
            ### if the image is colored then all channels must be included and a 3D kernel needs to be used.
            if len(padded_img.shape)==3:
                result[int((i-padding)/stride[0]),int((j-padding)/stride[1])]=np.sum(kernel*padded_img[i-kernel_size//2:i+(kernel_size//2)+1,j-kernel_size//2:j+(kernel_size//2)+1,:])
            else:   
                result[int((i-padding)/stride[0]),int((j-padding)/stride[1])]=np.sum(kernel*padded_img[i-kernel_size//2:i+(kernel_size//2)+1,j-kernel_size//2:j+(kernel_size//2)+1])
    ### return the result
    return result    

def Normalize(img:np.ndarray,min_val:np.float32=0.0,max_val:np.float32=1.0)->np.ndarray:
    '''
        Normalize function helps to get the image output in a range of [0,1].
        It is useful specically in the case of sharpening, as the output has negative elements

        Args:
            img (np.ndarray): The input image.
            min_val (np.float32): minimum value of pixel expected in output
            max_val (np.float32): maximum value of pixel expected in output
        Returns:
            np.ndarray: The normalized image
    '''
    min=np.min(img) # finds the minimum value in the image
    max=np.max(img) # finds the maximum value in the image
    min_img=np.ones(img.shape)*min
    min_val_img=np.ones(img.shape)*min_val
    img=min_val_img+(img-min_img)*((max_val-min_val)/(max-min))
    return img



def outConvPoolSize(imgSize:tuple,kernelSize:tuple,strideSize:tuple,paddingSize:tuple,poolSize:tuple,outChannels:np.uint):
    '''
        outConvPoolSize function helps to calculate the number of channels, image-size that should be given to the next convolutional layer in the CNN architecture as input

        Args:
            imgSize (tuple): image size inputted to convolutional layer
            kernelSize (tuple): convolutional kernel size
            strideSize (tuple): convolutional stride size
            paddingSize (tuple): convolutional padding size
            poolSize (tuple): pooling kernel size
            outChannels (np.uint):number of output channels needed after convolution
            
        Returns:
            outChannels: number of output channels after convolution
            (Nh,Nw): height and width of image output after convolution
            (int(Nh//poolSize[0]),int(Nw//poolSize[1])): height and width of image output after pooling
    '''
    Nh=int(np.floor((imgSize[0]+2*paddingSize[0]-kernelSize[0])/strideSize[0]))+1
    Nw=int(np.floor((imgSize[1]+2*paddingSize[1]-kernelSize[1])/strideSize[1]))+1
    return outChannels,(Nh,Nw),(int(Nh//poolSize[0]),int(Nw//poolSize[1]))

def fcInput(imgSize:tuple,inChannels:np.uint):
    '''
        fcInput function helps to calculate the vector input size to the fully connected layer input
        
        Args:
            imgSize (tuple): image size outputted from previous convolutional-pool layer
            inChannels: number of output channels from previous convolution-pool layer
        
        Returns:
            int(imgSize[0]*imgSize[1]*inChannels): vector input size to the fully connected layer input
    '''
    return int(imgSize[0]*imgSize[1]*inChannels)


#### Needs to be commented #####
# Custom Dataset 
class customDataset(Dataset):
    
    def __init__(self,tensors,transform=None):

        # check that size of data and labels match
        assert all(tensors[0].size(0)==t.size(0) for t in tensors),"Size mismatch between tensors"

        # assign inputs
        self.tensors=tensors
        self.transform=transform

    # getting an item from the dataset
    def __getitem__(self, index):

        # return transformed version of x if there are transforms
        if self.transform:
            x=self.transform(self.tensors[0][index])
        else:
            x=self.tensors[0][index]
        
        # return labels
        y=self.tensors[1][index]

        return x,y # return the (data,label) tuple
    
    def __len__(self):
        return self.tensors[0].size(0)



# L1 loss function
class L1loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        return tr.mean(tr.abs(yHat-y))
    
# L2 + Average loss function
class L2loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        return tr.mean((yHat-y)**2)+tr.abs(tr.mean(yHat))

# Cross-Correlatio loss function
class Corrloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        return -(tr.sum((yHat-tr.mean(yHat))*(y-tr.mean(y))))*(1/((yHat.numel()-1)*tr.std(yHat)*tr.std(y)))
    
# Variance loss function (Additional Explorations)
class VarLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        return tr.var(yHat)

# Weighted L2 & Average loss (Additional Exploration)
class WeightedL2loss(nn.Module):
    def __init__(self,a=0.9,b=0.1):
        super().__init__()
        self.a=a
        self.b=b

    def forward(self,yHat,y):
        return self.a*tr.mean((yHat-y)**2)+self.b*tr.abs(tr.mean(yHat))