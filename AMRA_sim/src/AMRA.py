import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import loadtxt
from pqdict import pqdict
'''
Author: Shusen Lin, Rohan Bosworth
This is the AMRA demo programm developed for ERL motion planning research purpose
This A* algorithm with multi-resolution and multi-heursitic strategies 

Update the Multiresolution - large map test needed
Need to update the multi-heuristic
'''
TEST_MAP = 2 #1->large map, 2->medium map, 3->small map
RES = 10 #largest resolution for search, 3**RES
EPSLION = 10 #weighted A* parameters
NORM = 1 # option for h value

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

class Env:
  def __init__(self,target_pos,envmap):
    self.goal_node = A_star.node(None,tuple(target_pos))
    self.map  = envmap

  def isGoal(self,input_node):
    pos_node = input_node.pos
    pos_goal = self.goal.pos
    return pos_node == pos_goal

  def getHeuristic(self,input_node,epslion):
    x_distance = np.abs(input_node.pos[0]-self.goal_node.pos[0])
    y_distance = np.abs(input_node.pos[1]-self.goal_node.pos[1])
    if NORM == 1:
      h_value = x_distance+y_distance# norm 1  
    elif NORM == 2:
      h_value = np.linalg.norm([x_distance,y_distance])

    return h_value*epslion

class A_star:
  '''
  def the class for A star algorithm
  '''
  def __init__(self,env,resolution=1):
    
    self.open_list = pqdict({})
    # self.envmap = env.map
    self. env = env
    self.envmap = self.env.map
    self.state_graph = {}
    len_row, len_col = self.env.map.shape
    for idx in range(len_row):
      for jdx in range(len_col):
        key = str((idx,jdx))
        self.state_graph[key] = None
    self.resolution = resolution

  class node:
    '''
    define the Atart node
    '''
    def __init__(self,parent=None,position=None):
      self.pos = position
      self.parent = parent
      self.g = np.inf
      self.h = 0
      self.f = np.inf

    def __str__(self):
      string = 'pos: ('+str(self.pos) + "), g: "+str(self.g)+ ", h: "+str(self.h) + ' '+str(self.parent)
      return string

    def __eq__(self, other):
      return self.pos == other.pos
 
  class state_space:
    '''
    define the state space
    '''
    def __init__(self,row,col) -> None:
        self.open_graph = pqdict({})

    def get(self,key):
      if self.open_graph[key] is None:
          return None
      status = self.open_graph[key]
      return status
    
  def motion_model(n=0):
    '''
    The motion model for multi-sacle features grid map, which depends on the power 3
    
    '''
    #1 is the lowest, anchor resolution. Greater values result in coarser resolutions
    scalar = 3**n
    motion = [[-scalar, 0], # move up
              [0 ,-scalar], # move left
              [scalar , 0], # move down
              [0 , scalar], # move right
              [-scalar, scalar], # move up-right
              [-scalar,-scalar], # move up-left
              [scalar ,-scalar], # move down-left
              [scalar , scalar]] # move down-right

    return motion

  def action_back(child_node,parent_node):
      if parent_node is not None:
        x_move = child_node.pos[0] - parent_node.pos[0]
        y_move = child_node.pos[1] - parent_node.pos[1]
      return (x_move,y_move)

  def recover_path(current_node):
    action = []
    while current_node.parent is not None:
      action.insert(0,A_star.action_back(current_node,current_node.parent))
      current_node = current_node.parent
    return action
  
  def check_window(self,center,resolution):
    '''
    The function to check the window whhether is occupied or free
    '''
    half = int(((3**resolution)-1)/2)
    rmin = center[0] - half
    rmax = center[0] + half
    cmin = center[1] - half
    cmax = center[1] + half
    check = self.envmap[rmin:rmax , cmin:cmax].sum()
    return check != 0 
  
  def stage_cost(self,start_pos,end_pos):
    return np.linalg.norm(start_pos-end_pos)
    # return 1

  def Go_Astar(self,start_pos,epsilon=1):
    '''
    The main loop to implement the Astar algorithm
    '''
    start_node = A_star.node(None,tuple(start_pos))
    start_node.g = 0
    start_node.h = self.env.getHeuristic(start_node,epsilon)
    start_node.f = start_node.g + start_node.h

    self.state_graph[str(tuple(start_pos))] = {'open?':True,'node':start_node}

    self.env.goal_node.g = self.env.goal_node.f = np.inf
    self.env.goal_node.h = self.env.getHeuristic(self.env.goal_node,epsilon)

    self.open_list[str(tuple(start_node.pos))] = start_node.f
    closed_list = []

    iteration = 0
    print('A-star calculating...')
    t0 = tic()

    while len(self.open_list) > 0:
        # while end_node not in closed_list:
        iteration+=1  
        # print(iteration)
        smallest_node_pos = self.open_list.popitem()[0]
        closed_list.append( self.state_graph[smallest_node_pos]['node'])
        self.state_graph[smallest_node_pos]['open?'] = False

        if self.state_graph[smallest_node_pos]['node'] ==  self.env.goal_node:
            print("A-star return!")
            toc(t0, nm="AMRA search")
            # print(self.state_graph[smallest_node_pos]['node'])
            return A_star.recover_path(self.state_graph[smallest_node_pos]['node']), self.state_graph
        
        for closed_node in closed_list:
          # for resolution_iter in reversed(range(self.resolution)):
          stop_multi_resoltion = False
          for resolution_iter in range(self.resolution):
            motion = A_star.motion_model(resolution_iter)
            for move in motion:
              # print('one move')
              x_pos = closed_node.pos[0] + move[0]
              y_pos = closed_node.pos[1] + move[1]
              closed_pos = np.array([closed_node.pos[0],closed_node.pos[1]])
              node_pos = np.array([x_pos,y_pos])
            
              # the following is to check the if the movment is valid
              if ( node_pos[0] < 0 or node_pos[0] >= self.envmap.shape[0] or node_pos[1] < 0 or node_pos[1] >= self.envmap.shape[1] ):
                # print('ERROR: out-of-map robot position commanded\n')
                if resolution_iter == 0:
                  continue
                else:
                  stop_multi_resoltion = True
                  break
              elif ( self.envmap[node_pos[0], node_pos[1]] != 0 ): #check the center point first 
                # print('ERROR: invalid robot position commanded\n')
                if resolution_iter == 0:
                  continue
                else:
                  stop_multi_resoltion = True
                  break
              elif self.check_window(node_pos,resolution_iter):#check whether the window is occupied 
                # print('ERROR: invalid window check position commanded\n')
                if resolution_iter == 0:
                  continue
                else:
                  stop_multi_resoltion = True
                  break
              elif self.state_graph[str(tuple(node_pos))] != None:
                if self.state_graph[str(tuple(node_pos))]['open?'] == False:
                  if resolution_iter == 0:
                    continue
                  else:
                    stop_multi_resoltion = True
                    break

              if self.state_graph[str(tuple(node_pos))] == None:
                  child  = A_star.node(closed_node,tuple(node_pos))
                  child.g = closed_node.g + self.stage_cost(closed_pos,node_pos)
                  child.h = self.env.getHeuristic(child,epsilon)
                  child.f = child.g + child.h
                  self.state_graph[str(tuple(node_pos))] = {'open?':True,'node':child}
                  self.open_list[str(tuple(node_pos))] = child.f

              elif (closed_node.g + self.stage_cost(closed_pos,node_pos) < self.state_graph[str(tuple(node_pos))]['node'].g):
                  self.state_graph[str(tuple(node_pos))]['node'].parent = closed_node
                  self.state_graph[str(tuple(node_pos))]['node'].g = closed_node.g + self.stage_cost(closed_pos,node_pos)
                  self.state_graph[str(tuple(node_pos))]['node'].f = self.state_graph[str(tuple(node_pos))]['node'].g + self.state_graph[str(tuple(node_pos))]['node'].h
                  self.state_graph[str(tuple(node_pos))]['open?'] = True

            if stop_multi_resoltion and resolution_iter!= 0:
              break
        closed_list.pop(0)

if __name__ == '__main__':

    if TEST_MAP == 1: # large map
      envmap = loadtxt('maps/map7.txt')
      robotstart = np.array([1, 1])
      targetstart = np.array([4998, 4998])
    elif TEST_MAP == 2:# medium map
      envmap = loadtxt('maps/map1.txt')
      robotstart = np.array([249, 1199])
      targetstart = np.array([1649, 1899])
    elif TEST_MAP == 3:#small map
      envmap = loadtxt('maps/map3.txt')
      #test map3 case 1
      robotstart = np.array([249, 249])
      targetstart = np.array([399, 399])
      # #test map3 case 2
      robotstart = np.array([74, 249])
      targetstart = np.array([399, 399])
      # # #test map3 case 3
      robotstart = np.array([0, 0])
      targetstart = np.array([399, 399])
    
    env = Env(targetstart,envmap)
    Astar = A_star(env,resolution=RES)
    path,graph = Astar.Go_Astar(robotstart,epsilon=EPSLION)

    now_pos = robotstart
    img = envmap
    for idx in range(envmap.shape[0]):
      for jdx in range(envmap.shape[1]):
        key = str(tuple([idx,jdx]))
        if graph[key] is not None:
          img[idx,jdx] = 50
    for action in path:
      next_pos_x = action[0]+now_pos[0]
      next_pos_y = action[1]+now_pos[1]
      img[next_pos_x,next_pos_y] = -55
      now_pos = (next_pos_x,next_pos_y)

    img[robotstart[0],robotstart[1]] = 70
    img[targetstart[0],targetstart[1]] = -70  
    img = np.where(img ==1 , -120, img)
    img = np.rot90(img,k=1,axes=(0,1))

    plt.imshow(img,cmap = 'jet')
    plt.imsave('test_map4.png',img, dpi = 1000,cmap = 'jet')
