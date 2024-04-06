import torch as tr

### Intersection Over Union

### intersection over union
def intersection_over_union(box_preds,ground_truth,box_format="corners",epsilon=1e-6):
    if box_format=="midpoint":
        box1_x1=box_preds[...,0:1]-box_preds[...,2:3] / 2
        box1_y1=box_preds[...,1:2]-box_preds[...,3:4] / 2
        box1_x2=box_preds[...,0:1]+box_preds[...,2:3] / 2
        box1_y2=box_preds[...,1:2]+box_preds[...,3:4] / 2

        box2_x1=ground_truth[...,0:1]-ground_truth[...,2:3] / 2
        box2_y1=ground_truth[...,1:2]-ground_truth[...,3:4] / 2
        box2_x2=ground_truth[...,0:1]+ground_truth[...,2:3] / 2
        box2_y2=ground_truth[...,1:2]+ground_truth[...,3:4] / 2
    
    else:
        box1_x1=box_preds[...,0:1]
        box1_y1=box_preds[...,1:2]
        box1_x2=box_preds[...,2:3]
        box1_y2=box_preds[...,3:4]

        box2_x1=ground_truth[...,0:1]
        box2_y1=ground_truth[...,1:2]
        box2_x2=ground_truth[...,2:3]
        box2_y2=ground_truth[...,3:4]

    x1=tr.max(box1_x1,box2_x1)
    y1=tr.max(box1_y1,box2_y1)
    x2=tr.min(box1_x2,box2_x2)
    y2=tr.min(box1_y2,box2_y2)

    # intersection (clamp(0) ensures the intersection is 0 if there is no overlap)
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)

    # computing box1 & box2 areas
    box1_area=abs((box1_x1-box1_x2)*(box1_y1-box1_y2))
    box2_area=abs((box2_x1-box2_x2)*(box2_y1-box2_y2))

    return intersection/(box1_area+box2_area-intersection+epsilon)

### Non Max Suppression (NMS)

def nms(predictions,confidence_threshold,iou_threshold,box_format="corners"):
    # predictions = [[class, probability_of_bbox, x1,y1,x2,y2],[],[]]
    assert type(predictions)==list
    bboxes=[box for box in predictions if box[1]>confidence_threshold]

    # sorting bounding with highest probability in the beganing
    bboxes=sorted(bboxes,lambda x:x[1],reverse=True)
    
    # creating list to store bounding boxes
    bboxes_after_nms=[]

    # main loop 
    while bboxes:
        chosen_box=bboxes.pop(0)
        bboxes=[
            box 
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(tr.tensor(chosen_box[2:]),
                                       tr.tensor(box[2:]),
                                       box_format=box_format
                                       )
            < iou_threshold 
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

