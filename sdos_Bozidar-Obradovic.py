import numpy as np
import matplotlib.pyplot as plt

import cv2
import scipy.ndimage as ndi

plt.close('all')
cv2.destroyAllWindows()

#%% functions

def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
        
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:

		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]
	
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick].astype("int")


def mask3(mask):
    m, n = np.shape(mask)
    mask_3d = np.zeros((m, n, 3))

    for i in range(3): 
        mask_3d[:, :, i] = mask
    return mask_3d


def players(mask, N1):
    connectivity = 8
    num, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(np.uint8(mask),\
                                          connectivity, cv2.CV_32S)
    area = stats[1:num, 4]
    humans = area.argsort()[-4:]
    
    mm = np.ones((N1,3))
    mm[:, 0:2] = centroids[humans+1]
    mm[:, 1] = mm[:, 1] + 25
    
    return mm


def court_extraction(frame):
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H = img[:, :, 0]
    
    N, bin_edges = np.histogram(H.ravel(), bins=180, range=[0, 180])
    ind = np.argmax(N)
    thresh = bin_edges[ind]   
    
    val1 = thresh - 7
    val2 = thresh + 7
    
    condition = (H > val1) & (H < val2)
    mask = np.where(condition, 1, 0)
    #scoreboard mask
    mask[620:690, 80:1120] = 0
    
    struct = ndi.generate_binary_structure(2, 2)
    mask = ndi.binary_opening(mask, structure = struct, iterations=10)
    
    struct = ndi.generate_binary_structure(2, 2)
    mask = ndi.binary_closing(mask, structure = struct, iterations=80)
    
    struct = np.zeros((3,3))
    struct[0, :] = 1
    mask = ndi.binary_dilation(mask, structure = struct, iterations=20)
    
    mask_3d = mask3(mask)
    show = (frame * mask_3d).astype(np.uint8)
    #scoreboard mask
    show[620:685, 80:1200, :] = 0

    return show


def player_detection(show):
    
    win_size = (48, 96)
    block_size = (16,16)
    block_stride = (8,8)
    cell_size = (8,8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, num_bins)
    hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    
    
    (rects, weights) = hog.detectMultiScale(show, winStride=(4, 4),
    		padding=(8, 8), scale=1.01)
            
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression_fast(rects, overlapThresh=0.3)
    
    del_box = []
    j = 0
    for(xA, yA, xB, yB) in pick:
        if yB - yA > 160:
            del_box.append(j)
        j = j + 1       
    pick_red = np.delete(pick, del_box, 0)
    
    return pick_red


def colour_detection(show, pick_red):
    
    test = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)
    
    #court has low hue value
    cond_court = test[:, :, 0] > 40  
    #removing NCAA logo colour, black and purple
    cond_black = test[:, :, 2] > 0
    cond_ncaa = (test[:, :, 1] < 160) & (test[:, :, 0] > 0)
    
    mask_arena1 = np.where(cond_court * cond_black * cond_ncaa, 1, 0)
    mask_arena = mask3(mask_arena1)
    
    show = (show * mask_arena).astype(np.uint8)
    img = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)
    
    histV = np.zeros(np.max(np.shape(pick_red)),)
    c = 0
    for(xA, yA, xB, yB) in pick_red:
        roi = img[yA:yB, xA:xB, :]
        V = roi[:, :, 2]
        N, bin_edges = np.histogram(V.ravel(), bins=256)
        peakV = np.argmax(N[1:])
        histV[c] = peakV
        c = c + 1
    
    lib_c = np.mean(histV[histV < 100])
    ok_c = np.mean(histV[histV > 200])
    
    cond_lib = (img[:,:, 2] > lib_c - 7) & (img[:,:, 2] < lib_c + 7)
    cond_ok = (img[:,:, 2] > ok_c - 50) & (img[:,:, 2] < ok_c + 7)
    
    mask_lib = np.where(cond_lib, mask_arena1, 0)
    mask_ok = np.where(cond_ok, mask_arena1, 0)
    
    struct = ndi.generate_binary_structure(2, 2)
    mask = ndi.binary_closing(mask_lib, structure = struct, iterations=5)
    
    struct = ndi.generate_binary_structure(2, 1)
    mask = ndi.binary_opening(mask, structure = struct, iterations=5)
    
    LIB = players(mask, 4)
    
    struct = ndi.generate_binary_structure(2, 1)
    mask = ndi.binary_opening(mask_ok, structure = struct, iterations=3)
    
    struct = ndi.generate_binary_structure(2, 2)
    mask = ndi.binary_closing(mask, structure = struct, iterations=10)
    
    OK = players(mask, 4)
    
    return LIB, OK


# def map_players(data1, data2, LIB, OK):
    
#     # H, _= cv2.findHomography(data1, data2, cv2.RANSAC, 5.0)
    
#     liblib = np.zeros((3,4))
#     c = 0
#     for (x,y,k) in LIB:
#         vals = np.array([[x],[y],[k]])
#         arr = np.matmul(H, vals)
#         liblib[:, c] = arr.ravel() / arr[2]
#         c = c + 1
#     lib_lib = np.transpose(liblib[0:2, :])

#     okok = np.zeros((3,5))
#     c = 0
#     for (x,y,k) in OK:
#         vals = np.array([[x],[y],[k]])
#         arr = np.matmul(H, vals)
#         okok[:, c] = arr.ravel() / arr[2]
#         c = c + 1
    
#     ok_ok = np.transpose(okok[0:2, :])
    
#     return lib_lib, ok_ok

def map_players(H, LIB, OK):
    
    # H, _= cv2.findHomography(data1, data2, cv2.RANSAC, 5.0)
    
    liblib = np.zeros((3,4))
    c = 0
    for (x,y,k) in LIB:
        vals = np.array([[x],[y],[k]])
        arr = np.matmul(H, vals)
        liblib[:, c] = arr.ravel() / arr[2]
        c = c + 1
    lib_lib = np.transpose(liblib[0:2, :])

    okok = np.zeros((3,5))
    c = 0
    for (x,y,k) in OK:
        vals = np.array([[x],[y],[k]])
        arr = np.matmul(H, vals)
        okok[:, c] = arr.ravel() / arr[2]
        c = c + 1
    
    ok_ok = np.transpose(okok[0:2, :])
    
    return lib_lib, ok_ok

#%% main

cap = cv2.VideoCapture('clipping.mp4')

data1 = np.array([[427,358],[317,427],[569,370],[467,447],\
                  [781,392],[711,470], [563,278],[54,590]])
    
data2 = np.array([[6,175],[6,330],[96,175],[96,330],\
                  [190,175],[190,330],[6,16],[6,490]])
H, _= cv2.findHomography(data1, data2, cv2.RANSAC, 5.0)

court = cv2.imread('court.jpg')

im_vid = []
ret, frame = cap.read()
while ret:
    teren = cv2.imread('court.jpg')
    frame = frame[:, 0:1120, :]
    
    show = court_extraction(frame)
    pick_red = player_detection(show)
    LIB, OK = colour_detection(show, pick_red)
    lib, ok = map_players(H, LIB, OK)
    
    for (x,y) in lib:
        cv2.circle(teren, (int(x),int(y)), 5, (0,0,255), -1)
        cv2.circle(court, (int(x),int(y)), 5, (0,0,255), -1)
    for (x,y) in ok:
        cv2.circle(teren, (int(x),int(y)), 5, (0,255,0), -1)
        cv2.circle(court, (int(x),int(y)), 5, (0,255,0), -1)
    
    im_vid.append(teren)
    ret, frame = cap.read()

(y,x,_) = im_vid[0].shape
vid = cv2.VideoWriter('track.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (x,y))

for i in range(len(im_vid)):
    vid.write(im_vid[i])
    
vid.release()
cap.release()
 






