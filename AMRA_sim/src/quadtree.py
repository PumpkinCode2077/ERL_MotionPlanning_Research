import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

class Node:
    def __init__(self, idx, rmin, rmax, cmin, cmax, map, depth_max, resolution):
        '''
        The main quadtree node, each node has four default children node
        
        
        '''
        self.bounds = (rmin, rmax, cmin, cmax)
        self.idx = idx
        
        depth_flag = len(idx) <= depth_max
        
        occ_sum = map[rmin:rmax-1 , cmin:cmax-1].sum()

        # determin whether children is all empty
        occ_empty = (occ_sum > 0)
        # determin whether children is all occupied
        occ_full = (occ_sum != (rmax-rmin-1)*(cmax-cmin-1))
        occ_grid_flag = occ_empty and occ_full

        res_flag = ((cmax-cmin)//2)>resolution and ((rmax-rmin)//2)>resolution

        row_2, col_2 = (rmax-rmin)//2, (cmax-cmin)//2

        valid_child = occ_grid_flag and res_flag and depth_flag

        if valid_child:
            self.children = ( Node(idx+'0', rmin,rmin+row_2, cmin,cmin+col_2, map, depth_max, resolution),
                              Node(idx+'1', rmin,rmin+row_2, cmin+col_2,cmax, map, depth_max, resolution),
                              Node(idx+'2', rmin+row_2,rmax, cmin,cmin+col_2, map, depth_max, resolution),
                              Node(idx+'3', rmin+row_2,rmax, cmin+col_2,cmax, map, depth_max, resolution))
        else:
            self.children = ()

class QuadTree:
    '''
    The main class to implement the quadtree algorithm
    
    '''
    def __init__(self, map, depth_max=10, resolution=1) -> None:

        self.depth_max = depth_max
        self.resolution = resolution
        self.map = map

        height,width = map.shape
        self.node = Node('0', 0, height, 0, width, self.map, depth_max, resolution)
    
######################################################################################################################
# For visualization

def extract_end_node (node):
    '''
    '''
    end_nodes = []
    
    if len(node.children) == 0:
        end_nodes.append(node)
    else:
        for n in node.children:
            end_nodes.extend( extract_end_node(n) )

    return end_nodes
    
################################################################################
def bounds2lines (bounds):
    '''
    takes the boundary of a rectangle in column-row coordinates
    returns line-segments (pairs of points) in x-y coordinates
    '''
    [rmin,rmax , cmin,cmax] = bounds
    return (((cmin, rmin), (cmin, rmax)),
            ((cmin, rmax), (cmax, rmax)),
            ((cmin, rmin), (cmax, rmin)),
            ((cmax, rmin), (cmax, rmax)))

########## load map
envmap = np.loadtxt('maps/map3.txt')
plt.imsave('test.png',envmap, dpi = 500,cmap = 'Greys')

filename = 'test.png'

img = cv2.imread( filename, cv2.IMREAD_GRAYSCALE) 

tic = time.time()

qt = QuadTree(envmap, depth_max=30, resolution=1)

my_map = qt.img_bin
# qt = QuadTree(img, depth_max=10, resolution=10)
# qt = QuadTree(img, depth_max=9, resolution=1)
print('time to complete: {:.5f}'.format(time.time()-tic))
# lines = [bounds2lines(en.bounds) for en in extract_end_node(qt.node)]
# print('number of end nodes: {:d}'.format(len(lines)))

# since I just want the lines for plotting, use set to remove duplicates
lines = list(set([l for en in extract_end_node(qt.node) for l in bounds2lines(en.bounds)]))
print('number of end nodes: {:d}'.format(len(lines)))

if 1:
    max_size = 12
    h, w = img.shape
    H,W = [max_size, (float(w)/h)*max_size] if (h > w) else [(float(h)/w)*max_size, max_size]
    fig, axes = plt.subplots(1,1, figsize=(W, H))

    if 0:
        # plot lines with matplotlib - vector quality for print
        axes.imshow(img, cmap='gray', alpha=1, interpolation='nearest', origin='lower')
        for l in lines: axes.plot( [l[0][0], l[1][0]], [l[0][1], l[1][1]], 'b-')

    else:
        # plot lines with opencv - way faster
        img_cv = np.stack([img for _ in range(3)], axis=2)
        for l in lines: cv2.line(img_cv, l[0], l[1], (0,0,255), 1)
        axes.imshow(img_cv, alpha=1, interpolation='nearest', origin='lower')
                
    # axes.plot(X,Y, 'r,')

    axes.axis('off')
    plt.tight_layout()
    plt.show()
