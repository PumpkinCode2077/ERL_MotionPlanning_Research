import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import loadtxt
from pqdict import pqdict
import utils as tools
'''
Author: Shusen Lin, Vaibhav Bishi
This is the AMRA demo programm developed for ERL motion planning research purpose
This A* algorithm with multi-resolution and multi-heursitic strategies 

Update the Multiresolution - large map test needed

Adding the Multi-Heuristic features
'''
TEST_MAP =  3#1->large map, 2->medium map, 3->small map， 4->tiny map
RES = 1 #largest resolution for search, 3**RES
EPSLION = 3 #weighted A* parameters
NORM = 1 # option for h value
NH = 2 # number of heuristic
W1 = 1 # expanding speed
W2 = 1 # use of inadmissble heuristic
DF_COEFFICIENT_b = 20
RAD = 100


class Env:
  def __init__(self,robot_pos,target_pos,envmap):
    self.map  = envmap
    self.robot_start = robot_pos
    self.target_end  = target_pos
    self.straight_bad_points = self.generate_straight_bad()
    DF_COEFFICIENT_a = np.min(self.map.shape)
    # print(DF_COEFFICIENT_a)
    self.DF_COEFFICIENT  = DF_COEFFICIENT_a * DF_COEFFICIENT_b

  def isGoal(self,input_node):
    pos_node = input_node.pos
    pos_goal = self.goal.pos
    return pos_node == pos_goal

  def generate_straight_bad(self):
    '''
    The function for generating the straight bad points
    '''
    sx = self.robot_start[0]
    sy = self.robot_start[1]
    ex = self.target_end[0]
    ey = self.target_end[1]
    bresenham = tools.bresenham2D(sx,sy,ex,ey)
    bad_points = np.array([[0,0]])
    for idx in range(len(bresenham[0])):
      if self.map[bresenham[0][idx],bresenham[1][idx]] != 0:
        bad_points = np.append(bad_points,[[bresenham[0][idx],bresenham[1][idx]]],axis=0)
    return bad_points[1:]

  def straight_traj_check(self,input_node,norm2):
    x_pos = input_node.pos[0]
    y_pos = input_node.pos[1]
    current_pos = np.array([x_pos,y_pos])
    dist = np.linalg.norm((current_pos-self.straight_bad_points),axis=1)
    dist = np.min(dist)
    if dist == 0:
      return np.inf
    elif dist < RAD:
      value = 1/dist*self.DF_COEFFICIENT
      return value
    else:
      return norm2+1

  def get_multiHeuristic(self,input_node,nh):
    '''
    The function works for multi_heuristic
    nh = 0: norm 1, admissible
    nh = 1: norm 2, admissible
    nh = 2: straight check, inadmissble 

    input: the desired node for calculating the heuristic
    nh:    the desired heuristic selection
    '''
    x_distance = np.abs(input_node.pos[0]-self.target_end[0])
    y_distance = np.abs(input_node.pos[1]-self.target_end[1])
    if nh == 3:
      h_value = x_distance+y_distance# norm 1  
    elif nh == 0:
      h_value = np.linalg.norm([x_distance,y_distance])#norm 2
    elif nh == 1:
      norm2   = np.linalg.norm([x_distance,y_distance])
      h_value = self.straight_traj_check(input_node,norm2)
    return h_value

  def getHeuristic(self,input_node,epslion,nh): 
    return self.get_multiHeuristic(input_node,nh)*epslion

class A_star:
  '''
  def the class for A star algorithm
  '''
  def __init__(self,env,resolution=1):
    
    #we use two openlist here
    self.open_list0 = pqdict({})#for norm 1
    self.open_list1 = pqdict({})#for nomr 2
  
    self.open_list = [self.open_list0,self.open_list1]
    self.closed_list_anchor = []
    self.closed_list_inad = []


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
    define the MHRA node, a node has more than one values for g,h,f
    '''
    def __init__(self,parent=None,position=None):
      self.pos = position
      self.parent = parent
      self.g = np.inf
      self.h = [np.inf,np.inf]
      self.f = [np.inf,np.inf]

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

  def action_back(self,child_node,parent_node):
      if parent_node is not None:
        x_move = child_node.pos[0] - parent_node.pos[0]
        y_move = child_node.pos[1] - parent_node.pos[1]
      return (x_move,y_move)

  def recover_path(self,current_node):
    action = []
    while current_node.parent is not None:
      action.insert(0,self.action_back(current_node,current_node.parent))
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
    # return np.linalg.norm(start_pos-end_pos)
    return 1
  
  def Key(self,node_pos,i):
    key = self.state_graph[str(tuple(node_pos))]['node'].g + \
          W1*self.env.getHeuristic(self.state_graph[str(tuple(node_pos))]['node'],EPSLION,i)
    return key
  
  def ExpandState(self,current_pos):
    for hdx in range(NH):
      if(self.open_list[hdx]):
        try:
          self.open_list[hdx].pop(current_pos)  
        except:
          KeyError

    # print(len(self.open_list[0]))
    closed_node = self.state_graph[current_pos]['node']

    stop_multi_resoltion = False
    for resolution_iter in range(self.resolution):
      motion = A_star.motion_model(resolution_iter)
      for move in motion:
        # print('one move')
        x_pos = closed_node.pos[0] + move[0]
        y_pos = closed_node.pos[1] + move[1]
        closed_pos = np.array([closed_node.pos[0],closed_node.pos[1]])
        node_pos = np.array([x_pos,y_pos])
      
        # the following is to check the if the movment is valid**********************************************************
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
        # elif self.state_graph[str(tuple(node_pos))] != None:
        #   if self.state_graph[str(tuple(node_pos))]['open?'] == False:
        #     if resolution_iter == 0:
        #       continue
        #     else:
        #       stop_multi_resoltion = True
        #       break
        #***************************************************************************************************************
        if self.state_graph[str(tuple(node_pos))] == None:
            child  = A_star.node(closed_node,tuple(node_pos))
            self.state_graph[str(tuple(node_pos))] = {'open?':True,'node':child}
            self.state_graph[str(tuple(node_pos))]['node'].g = np.inf
            
        if self.state_graph[str(tuple(node_pos))]['node'].g > closed_node.g + self.stage_cost(closed_pos,node_pos):
          self.state_graph[str(tuple(node_pos))]['node'].g = closed_node.g + self.stage_cost(closed_pos,node_pos)
          self.state_graph[str(tuple(node_pos))]['node'].parent = closed_node
          self.state_graph[str(tuple(node_pos))]['open?'] = True
          
          if self.state_graph[str(tuple(node_pos))]['node'].pos not in self.closed_list_anchor:
            self.open_list[0][str(tuple(node_pos))] = self.Key(node_pos,0)
            if self.state_graph[str(tuple(node_pos))]['node'].pos not in self.closed_list_inad:
              for hdx in range(1,NH):
                if self.Key(node_pos,hdx) <= W2* self.Key(node_pos,0):
                  self.open_list[hdx][str(tuple(node_pos))] =  self.Key(node_pos,hdx)

          # if (self.state_graph[str(tuple(node_pos))]['node'].pos) == (29,59):
          #   print("AShit")

      if stop_multi_resoltion and resolution_iter!= 0:
        break

  def Go_Astar(self,start_pos,end_pos,epsilon=1):
    '''
    The main loop to implement the Astar algorithm
    '''
    #MHA initialization
    start_node = A_star.node(None,tuple(start_pos))
    start_node.g = 0
    start_node.h[0] = self.env.getHeuristic(start_node,epsilon,0)
    start_node.f[0] = start_node.g + start_node.h[0]
    start_node.h[1] = self.env.getHeuristic(start_node,epsilon,1)
    start_node.f[1] = start_node.g + start_node.h[1]

    end_node = A_star.node(None,tuple(end_pos))
    end_node.g = np.inf
    end_node.h[0] = 0
    end_node.f[0] = start_node.g + start_node.h[0]
    end_node.h[1] = 0
    end_node.f[1] = start_node.g + start_node.h[1]

    self.state_graph[str(tuple(start_pos))] = {'open?':True,'node':start_node}
    self.state_graph[str(tuple(end_pos))] = {'open?':False,'node':end_node}

    self.open_list[0][str(tuple(start_node.pos))] = start_node.f[0]
    self.open_list[1][str(tuple(start_node.pos))] = start_node.f[1]
    # print(self.open_list[0])
    # print(self.open_list[1])

    iteration = 0
    print('MHRA calculating...')
    t0 = tools.tic()

    while len(self.open_list[0]) > 0:
        iteration+=1  
        for hdx in range(1,NH):
          if self.open_list[hdx]:
            temp = self.open_list[hdx].topitem()[1]
          else:
            temp =np.inf
          if temp <= W2*self.open_list[0].topitem()[1]:
            if self.state_graph[str(tuple(end_pos))]['node'].g <= self.open_list[hdx].topitem()[1]:
              if self.state_graph[str(tuple(end_pos))]['node'].g < np.inf:
                print("MHRA end, return the path!")
                tools.toc(t0, nm="MHRA search")
                print('anchor closed list length: '+ str(len(self.closed_list_anchor)))
                print('inadmissible closed list length: '+ str(len(self.closed_list_inad)))
                return self.recover_path(self.state_graph[str(tuple(end_pos))]['node']), self.state_graph
              
            else:
              smallest_node_pos = self.open_list[hdx].topitem()[0]
              self.ExpandState(smallest_node_pos)
              self.closed_list_inad.append(self.state_graph[smallest_node_pos]['node'].pos)
              self.state_graph[smallest_node_pos]['open?'] = False

          else:
            if self.state_graph[str(tuple(end_pos))]['node'].g <= self.open_list[0].topitem()[1]:
              if self.state_graph[str(tuple(end_pos))]['node'].g < np.inf:
                print("MHRA end, return the path!")
                tools.toc(t0, nm="AMRA search")
                print('anchor closed list length: '+ str(len(self.closed_list_anchor)))
                print('inadmissible closed list length: '+ str(len(self.closed_list_inad)))
                return self.recover_path(self.state_graph[str(tuple(end_pos))]['node']), self.state_graph
            else:
              
              smallest_node_pos = self.open_list[0].topitem()[0]
              self.ExpandState(smallest_node_pos)
              # print(smallest_node_pos)
              self.closed_list_anchor.append(self.state_graph[smallest_node_pos]['node'].pos)
              # print(self.closed_list_anchor)
              self.state_graph[smallest_node_pos]['open?'] = False

    print('anchor closed_list length: '+ str(len(self.closed_list_anchor)))
    print('inadmissible closed_list length: '+ str(len(self.closed_list_inad)))
    return self.recover_path(self.state_graph[str(tuple(end_pos))]['node']), self.state_graph     

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
      # robotstart = np.array([0, 0])
      targetstart = np.array([399, 399])
    elif TEST_MAP == 4:#tiny map
      envmap = loadtxt('maps/map5.txt')
      robotstart = np.array([0, 0])
      targetstart = np.array([29, 59])
    
    env = Env(robotstart,targetstart,envmap)
    # print(env.generate_straight_bad())
    Astar = A_star(env,resolution=RES)
    path,graph = Astar.Go_Astar(robotstart,targetstart,epsilon=EPSLION)

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
