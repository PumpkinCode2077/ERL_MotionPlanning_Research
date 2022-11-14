import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import loadtxt
from pqdict import pqdict
'''
Author: Shusen Lin

This is experimental test code for A-star search, algorithm does apply the 
priority dictionary, for the challenge map7 may goes slow, run this code 
will return the search space, moving trajectory and searching time.

'''

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

    h_value = x_distance+y_distance# norm 1  
    # h_value = np.math.sqrt(x_distance^2 + y_distance^2)# norm 2

    return h_value*epslion

class A_star:
  '''
  def the class for A star algorithm
  '''
  def __init__(self,env,resolution=0):
    
    self.open_list = pqdict({})
    self.envmap = env.map
    self.state_graph = {}
    len_row, len_col = env.map.shape
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
      self.parent = []
      self.parent.append(parent)
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
    scalar = n ** 3
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
        move = (x_move,y_move)
        #TODO: Change action back for multiple resolutions
        motion = A_star.motion_model(1)
        for action in motion:
          if move == tuple(action):
            return tuple(action)
      return 'invalid move'

  def recover_path(current_node):
    action = []
    while current_node.parent[0] is not None:
      smallest_idx = 0
      smallest_g = current_node.parent[0].g
      for p_dx, p_node in enumerate(current_node.parent):
        if p_node.g < smallest_g:
          smallest_g = p_node.g
          smallest_idx = p_dx

      optimal_parent = current_node.parent[smallest_idx]
      # if optimal_parent is not None:
      action.insert(0,A_star.action_back(current_node,optimal_parent))
      current_node = optimal_parent

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
  
  def stage_cost(start_pos,end_pos):
    return np.linalg.norm(start_pos-end_pos)

  def Go_Astar(self,start_pos,epsilon=1):
    '''
    The main loop to implement the Astar algorithm
    '''
    start_node = A_star.node(None,tuple(start_pos))
    start_node.g = 0
    start_node.h = env.getHeuristic(start_node,epsilon)
    start_node.f = start_node.g + start_node.h

    self.state_graph[str(tuple(start_pos))] = {'open?':True,'node':start_node}

    env.goal_node.g = env.goal_node.f = np.inf
    env.goal_node.h = env.getHeuristic(env.goal_node,epsilon)

    self.open_list[str(tuple(start_node.pos))] = start_node.f
    closed_list = []

    iteration = 0
    motion = A_star.motion_model(1)
    print('A-star calculating...')

    while len(self.open_list) > 0:
        # while end_node not in closed_list:
        iteration+=1  
        print(iteration)
        smallest_node_pos = self.open_list.popitem()[0]
        closed_list.append( self.state_graph[smallest_node_pos]['node'])
        self.state_graph[smallest_node_pos]['open?'] = False

        if self.state_graph[smallest_node_pos]['node'] ==  env.goal_node:
            print("A-star return!")
            # return open_node
            return self.state_graph
            # return A_star.recover_path(self.state_graph[smallest_node_pos]['node']), self.state_graph
            # return state_graph

        stage_cost = 1 
        for closed_node in closed_list:
          for resolution_iter in reversed(range(self.resolution)):
            motion = A_star.motion_model(resolution_iter)
            for move in motion:
              # print('one move')
              x_pos = closed_node.pos[0] + move[0]
              y_pos = closed_node.pos[1] + move[1]
              closed_pos = 
              node_pos = np.array([x_pos,y_pos])
              # the following is to check the if the movment is valid
              if ( node_pos[0] < 0 or node_pos[0] >= self.envmap.shape[0] or node_pos[1] < 0 or node_pos[1] >= self.envmap.shape[1] ):
              # print('ERROR: out-of-map robot position commanded\n')
                if resolution_iter == 0:
                  continue
                else:
                  break
              elif ( self.envmap[node_pos[0], node_pos[1]] != 0 ): #check the center point first 
              # print('ERROR: invalid robot position commanded\n')
                if resolution_iter == 0:
                  continue
                else:
                  break
              elif self.check_window(node_pos,resolution_iter):#check whether the window is occupied 
                if resolution_iter == 0:
                  continue
                else:
                  break
              elif self.state_graph[str(node_pos)] != None:
                if self.state_graph[str(node_pos)]['open?'] == False:
                  if resolution_iter == 0:
                    continue
                  else:
                    break
              if self.state_graph[str(node_pos)] == None:
                  child  = A_star.node(closed_node,tuple(node_pos))
                  child.g = closed_node.g + stage_cost(closed_node.pos,node_pos)
                  child.h = env.getHeuristic(child,epsilon)
                  child.f = child.g + child.h
                  self.state_graph[str(node_pos)] = {'open?':True,'node':child}
                  self.open_list[str(node_pos)] = child.f

              elif (closed_node.g + stage_cost) < self.state_graph[str(node_pos)]['node'].g:
                  self.state_graph[str(node_pos)]['node'].parent = closed_node
                  self.state_graph[str(node_pos)]['node'].g = closed_node.g + stage_cost
                  self.state_graph[str(node_pos)]['node'].f = self.state_graph[str(node_pos)]['node'].g + self.state_graph[str(node_pos)]['node'].h
                  self.state_graph[str(node_pos)]['open?'] = True

          closed_list.pop(0)

if __name__ == '__main__':

    envmap = loadtxt('maps/map0.txt')
    robotstart = np.array((0, 2))
    targetstart = np.array((5, 3))

    envmap = loadtxt('maps/map2.txt')
    robotstart = np.array((0, 2))
    targetstart = np.array((7, 9))

    envmap = loadtxt('maps/map3.txt')
    #test map3 case 1
    robotstart = np.array([249, 249])
    targetstart = np.array([399, 399])
    # #test map3 case 2
    robotstart = np.array([74, 249])
    targetstart = np.array([399, 399])
    # # #test map3 case 3
    robotstart = np.array([4, 399])
    targetstart = np.array([399, 399])

    # envmap = loadtxt('maps/map4.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([5, 6])

    # envmap = loadtxt('maps/map5.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 59])
    
    # envmap = loadtxt('maps/map6.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 36])

    # envmap = loadtxt('maps/map1.txt')
    # robotstart = np.array([249, 1199])
    # targetstart = np.array([1649, 1899])

    # envmap = loadtxt('maps/map7.txt')
    # robotstart = np.array([1, 1])
    # targetstart = np.array([4998, 4998])

    env = Env(targetstart,envmap)
    t0 = tic()
    
    Astar = A_star(env,resolution=4)

    path,graph = Astar.Go_Astar(env,robotstart,epsilon=3)

    # graph = Astar.Go_Astar(robotstart,epsilon=3)
    
    # graph = Astar.Go_Astar(env,robotstart,epsilon=3)

    toc(t0,'A-star get_path')
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
