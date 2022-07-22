import re
import os
from this import d

from sklearn.preprocessing import scale
import settings
import numpy as np
import SimpleITK as sitk

from functools import reduce
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

brains_dir = os.path.join(settings.data_dir, settings.net1data_dname,"brains")

def compose(*funcs):
        # return lambda x: reduce(lambda v, f: f(v), funcs, x)
        if funcs: return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
        else: raise ValueError("Composition of empty sequence not supported.")

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, 
                    jitter=0.3, hue=0.1, sat=1.5, val=1.5, proc_img=True):
        
        filename = annotation_line.split(',')[0].replace("labels","t1")
        
        box = np.array([int(val) for val in annotation_line.split(',')[1:]])
        box = box.reshape(1,box.shape[0])

        box_data = np.zeros((max_boxes, 7))
        
        if len(box) > 0:
                #np.random.shuffle(box)
                if len(box) > max_boxes:
                        box = box[:max_boxes]
                box_data[: len(box)] = box       
           
        image = sitk.ReadImage(os.path.join(brains_dir, filename), imageIO=settings.imgio_type)
        # Images normalized in a previous step.
        # If not, divide image_data by 255
        image_data = sitk.GetArrayFromImage(image)
                
        """ 
        # Dimensions (w,h,d)_img are dimensions of the 
        # loaded image, and h, w, d input dimensions for network
        w_img, h_img, d_img = image.GetSize()
        h , w , d  = input_shape
        
        # TODO: Incomplete. Step skipped because Images are resized
        # prior to augmentation. can proceed without scaling images
        scale = min(w/w_img, h/h_img, d/d_img)
                
        if [w_img, h_img, d_img] == [w, h , d]: 
                w_scaled, h_scaled, d_scaled = [w_img, h_img, d_img] 
        else:
                
                #for image
                w_scaled, h_scaled, d_scaled = [w_img * scale, h_img * scale, d_img * scale]
                w_diff, h_diff, d_diff = (np.array([w, h, d])-np.array([w_scaled, h_scaled, d_scaled]))//2
                #resize image
                
                #for bounding box
                box[0] = box[0] * scale + w_diff 
                box[1] = box[1] * scale + h_diff 
                box[2] = box[2] * scale + d_diff
        """
        image_data = np.expand_dims(image_data, axis=3) 
        
        return image_data, box_data   
        
# Prior to major modifications, function is refferred to
# as "preprocess_true_boxes" in original reference code 
def set_input_format(true_boxes, input_shape, anchors, no_classes):
        
        assert (true_boxes[..., 6] <= no_classes).all(), "class id must be less than num_classes"
        no_layers = len(anchors) // 3  # default setting
        
        anchor_mask = ([[6, 7, 8], [3, 4, 5], [0, 1, 2]] if no_layers == 3 else 
                       [[3, 4, 5], [1, 2, 3]])
        
        true_boxes  = np.array(true_boxes, dtype="float32")
        input_shape = np.array(input_shape, dtype="int32")
        
        boxes_ctr = (true_boxes[..., 0:3] + true_boxes[..., 3:6]) // 2
        boxes_whd = true_boxes[... , 3:6] - true_boxes[..., 0:3]
        
        true_boxes[..., 0:3] = boxes_ctr / input_shape[::-1]
        true_boxes[..., 3:6] = boxes_whd / input_shape[::-1]
        
        m = true_boxes.shape[0]
        
        grid_shapes = [input_shape // {0: 16, 1: 8, 2: 4}[l] for l in range(no_layers)]
        
        y_true = [ np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], grid_shapes[l][2], 
                             len(anchor_mask[l]), no_classes+7), dtype="float32") for l in range(no_layers)]
        
        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_ctrs    = anchors / 2.0
        anchor_invctrs = -anchor_ctrs
        
        # Valid anchors masks
        valid_mask = boxes_whd[..., 0] > 0
       
        for b in range(m):
                
            # Discard zeros
            whd = boxes_whd[b, valid_mask[b]]
            #print("--",m,b,valid_mask[b].shape,boxes_whd[b, valid_mask[b]].shape)#, boxes_whd[b, valid_mask[b]].shape)
            
            if len(whd) == 0:
                continue    
            # Expand dim to apply broadcasting.
            whd = np.expand_dims(whd, -2)
            
            
            box_ctrs    = whd / 2.0
            box_invctrs = -box_ctrs
        
            intersect_mins = np.maximum(box_invctrs, anchor_invctrs)
            intersect_maxs = np.minimum(box_ctrs, anchor_ctrs)
            
            intersect_whd = np.maximum(intersect_maxs  - intersect_mins, 0.0)
            intersect_area = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
                
            box_area    = whd[..., 0] * whd[..., 1] * whd[..., 2]
            anchor_area = anchors[..., 0] * anchors[..., 1] * anchors[..., 2]
            
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box           
            best_anchor = np.argmax(iou, axis=-1)
            
            for t, n in enumerate(best_anchor):
                    for l in range(no_layers):
                            if n in anchor_mask[l]:
                                    
                                    # b - iterates over images in batch, 0,1,2 - layers
                                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][0]).astype("int32")
                                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][1]).astype("int32")
                                    k = np.floor(true_boxes[b, t, 2] * grid_shapes[l][2]).astype("int32")
                                    
                                    a = anchor_mask[l].index(n)
                                    c = true_boxes[b, t, 6].astype("int32")
                                    
                                    y_true[l][b, k, j, i, a, 0:6] = true_boxes[b, t, 0:6]
                                    y_true[l][b, k, j, i, a, 6] = 1
                                    y_true[l][b, k, j, i, a, 7 + (c-1)] = 1 
        return y_true
                                        
        
def data_generator(annotation_lines, batch_size, input_shape, anchors, no_classes):

        n = len(annotation_lines)
        i = 0 ; c =0
        while True:
                if c ==5: break
                else: print("--",c)
                
                image_data = [] ; box_data = []
                for b in range(batch_size):
                    #if i == 0:
                    #       np.random.shuffle(annotation_lines)
                        
                    image, box = get_random_data(annotation_lines[i], input_shape)
                        
                    image_data.append(image)
                    box_data.append(box)
                    i = (i + 1) % n
                      
                image_data = np.array(image_data)
                box_data = np.array(box_data)
                     
                y_true = set_input_format(box_data, input_shape, anchors, no_classes)
                yield [image_data, *y_true], np.zeros(batch_size)
                
                c+=1
        
       
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, no_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0:
                return None
        return data_generator(annotation_lines, batch_size, input_shape, anchors, no_classes)

