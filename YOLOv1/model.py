import torch as tr
import torch.nn as nn
from torchsummary import summary

architecture_config=[
    #Tuple: (kernel_size, output_channels, stride, padding)
    (7,64,2,3),
    "M", # MAXPOOL 
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    # List: [Tuple1, Tuple2, no_of_times_tuple_repeats_in_sequence]
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        # super(CNNBlock,self).__init__()
        super().__init__() # both the implementations are same
        
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm=nn.BatchNorm2d(out_channels)
        self.leakyrelu=nn.LeakyReLU(0.1)
    
    def forward(self,x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self,in_channels=3, **kwargs):
        # super(YOLOv1,self).__init__()
        super().__init__()
        self.architecture=architecture_config
        self.in_channels=in_channels
        # YOLO call their conv-layers as darknet
        self.darknet=self._create_conv_layers(self.architecture)
        self.fcs=self._create_fcs(**kwargs)

    def forward(self,x):
        x=self.darknet(x)
        # return x
        return self.fcs(tr.flatten(x,start_dim=1))

    # create the darknet
    def _create_conv_layers(self,architecture):
        layers=[]
        in_channels=self.in_channels

        for x in architecture:
            if type(x)==tuple:
                layers+=[CNNBlock(
                    in_channels,
                    out_channels=x[1],
                    kernel_size=x[0],
                    stride=x[2],
                    padding=x[3]
                )]
                in_channels=x[1]

            elif type(x)==str:
                layers+=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
            elif type(x)==list:
                conv1 = x[0] # tuple
                conv2=x[1]   # tuple
                num_repeat=x[2] # integer

                for _ in range(num_repeat):
                    layers+=[
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ] 
                    layers+=[
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ] 
                    in_channels=conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self,split_size,num_boxes,num_classes):
        S,B,C=split_size,num_boxes,num_classes
        nodes=496
        fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, nodes), # original paper it is 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(nodes,S*S*(C+B*5)) # to be reshaped to (S,S,30) (20)+(2)*5=30
        )
        return fc_layers
    
def test(split_size=7,num_boxes=2,num_classes=20):
    model=YOLOv1(split_size=split_size,num_boxes=num_boxes,num_classes=num_classes)
    x=tr.randn((2,3,448,448))
    # summary(model,(3,448,448))
    # print(model(x).shape) 

# test()