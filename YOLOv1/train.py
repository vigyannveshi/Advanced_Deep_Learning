import torch as tr
import torchvision.transforms as T
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import YOLOv1
from dataset import VOCDataset 
from utils import intersection_over_union,non_max_suppression,mean_average_precision,cellboxes_to_boxes,get_bboxes,plot_image,save_checkpoint,load_checkpoint
from loss import YoloLoss

### Ensuring re-creation 
seed=1423
tr.manual_seed(seed)

### Hyperparameters etc.
LEARNING_RATE=2e-5
DEVICE="cuda" if tr.cuda.is_available() else "cpu"
BATCH_SIZE=16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS=2
PIN_MEMORY=True
LOAD_MODEL=False
LOAD_MODEL_FILE="overfit.pth.tar"
IMG_DIR="YOLOv1/data/images"
LABEL_DIR="YOLOv1/data/labels"

### Creating transforms
class Compose:
    def __init__(self,transforms):
        self.transforms=transforms
    
    def __call__(self,img,bboxes):
        for t in self.transforms:
            # transform only the image (resize)
            img,bboxes=t(img),bboxes
        return img,bboxes

transform=Compose([T.Resize((448,448)),T.ToTensor()])

def train_fn(train_loader,model,optimizer,lossfun):
    loop=tqdm(train_loader,leave=True)
    mean_loss=[]
    model.train()
    for batch_idx, (X,y) in enumerate(loop):
        # sending data to GPU
        X,y=X.to(DEVICE),y.to(DEVICE)
        yHat=model(X)
        loss=lossfun(yHat,y)
        mean_loss.append(loss.item())

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar 
        loop.set_postfix(loss=loss.item())
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model=YOLOv1(split_size=7,num_boxes=2,num_classes=20).to(DEVICE)
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    lossfun=YoloLoss()

    # if we need to load the model
    if LOAD_MODEL:
        load_checkpoint(tr.load(LOAD_MODEL_FILE),model,optimizer)


    train_dataset=VOCDataset(
        "YOLOv1/data/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    test_dataset=VOCDataset(
        "YOLOv1/data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        pred_boxes,target_boxes=get_bboxes(
            train_loader,model,iou_threshold=0.5,threshold=0.4
        )
        map=mean_average_precision(pred_boxes,target_boxes,iou_threshold=0.5,box_format="midpoint")
        train_fn(train_loader,model,optimizer,lossfun)
        print(f"{epoch+1}/{EPOCHS} complete with Train mAP: {map}")

if __name__ == "__main__":
    main()