import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import loadtxt
from pqdict import pqdict
'''
This is experimental test code for A-star search, it can only implement on 
the statical target, since this algorithm does not apply the priority dictionary 
yet, for large map like map3b, map3c, map1 and map7 may goes very slow, run this code 
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
  class node:
    def __init__(self,parent=None,position=None):
      self.pos = position
      self.parent = []
      self.parent.append(parent)
      self.g = np.inf
      self.h = 0
      self.f = np.inf
      self.open_list = pqdict({})
      
    def __str__(self):
      string = 'pos: ('+str(self.pos) + "), g: "+str(self.g)+ ", h: "+str(self.h) + ' '+str(self.parent)
      return string

    def __eq__(self, other):
      return self.pos == other.pos
 

  class state_space:
    def __init__(self,row,col) -> None:
        self.open_graph = pqdict({})

    def get(self,key):
      if self.open_graph[key] is None:
          return None
      status = self.open_graph[key]
      return status
    
  def motion_model():
    motion = [[-1, 0], # move up
              [0 ,-1], # move left
              [1 , 0], # move down
              [0 , 1], # move right
              [-1, 1], # move up-right
              [-1,-1], # move up-left
              [1 ,-1], # move down-left
              [1 , 1]] # move down-right

    return motion

  def action_back(child_node,parent_node):
      if parent_node is not None:
        x_move = child_node.pos[0] - parent_node.pos[0]
        y_move = child_node.pos[1] - parent_node.pos[1]
        move = (x_move,y_move)
        motion = A_star.motion_model()
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

  def Go_Astar(env,start_pos,epsilon=1):
    envmap = env.map
    state_graph = {}
    len_row, len_col = env.map.shape
    for idx in range(len_row):
      for jdx in range(len_col):
        key = str((idx,jdx))
        state_graph[key] = None

    start_node = A_star.node(None,tuple(start_pos))
    start_node.g = 0
    start_node.h = env.getHeuristic(start_node,epsilon)
    start_node.f = start_node.g + start_node.h
    env.goal_node.g = env.goal_node.f = np.inf
    env.goal_node.h = env.getHeuristic(env.goal_node,epsilon)

    open_list = [start_node]
    closed_list = []

    iteration = 0
    len_row, len_col = env.map.shape
    motion = A_star.motion_model()

    while len(open_list) > 0:
        # while end_node not in closed_list:
        iteration+=1  
        print('iteration in: '+str(iteration))
        smallest_node = open_list[0]
        
        for open_node in open_list:  #find the smallest f value node
          state_graph[str(open_node.pos)] = {'open?':True,'node':open_node}

          if open_node.f <= smallest_node.f:
              smallest_node = open_node

        print(smallest_node.pos)
    
        for o_idx, open_node in enumerate(open_list):#remove th samllest f value node from open to closed
          if open_node.f == smallest_node.f:
              closed_list.append(open_node)
              open_list.pop(o_idx)

              state_graph[str(open_node.pos)] = {'open?':False,'node':open_node}

              if open_node ==  env.goal_node:
                  # return open_node
                  return A_star.recover_path(open_node),state_graph

        stage_cost = 1 
        for closed_node in closed_list:
          for move in motion:
            x_pos = closed_node.pos[0] + move[0]
            y_pos = closed_node.pos[1] + move[1]
            node_pos = (x_pos,y_pos)
            # the following is to check the if the movment is valid
            if ( node_pos[0] < 0 or node_pos[0] >= envmap.shape[0] or node_pos[1] < 0 or node_pos[1] >= envmap.shape[1] ):
            # print('ERROR: out-of-map robot position commanded\n')
              continue
            elif ( envmap[node_pos[0], node_pos[1]] != 0 ):
            # print('ERROR: invalid robot position commanded\n')
              continue
            elif (abs(node_pos[0]-node_pos[0]) > 1 or abs(node_pos[1]-node_pos[1]) > 1):
            # print('ERROR: invalid robot move commanded\n')
              continue
            elif state_graph[str(node_pos)] != None:
              if state_graph[str(node_pos)]['open?'] == False:
                continue

            if state_graph[str(node_pos)] == None:
              child = A_star.node(closed_node,tuple(node_pos))
              child.g = closed_node.g + stage_cost
              child.h = env.getHeuristic(child,epsilon)
              child.f = child.g + child.h
              state_graph[str(node_pos)] = {'open?':True,'node':child}
              open_list.append(state_graph[str(node_pos)]['node'])

            else:
              state_graph[str(node_pos)]['node'].parent.append(closed_node)
              state_graph[str(node_pos)]['node'].g = min((closed_node.g + stage_cost),state_graph[str(node_pos)]['node'].g)
              state_graph[str(node_pos)]['node'].f = state_graph[str(node_pos)]['node'].g + state_graph[str(node_pos)]['node'].h
              state_graph[str(node_pos)]['open?'] = True

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
    #test map3 case 2
    robotstart = np.array([74, 249])
    targetstart = np.array([399, 399])
    #test map3 case 3
    robotstart = np.array([4, 399])
    targetstart = np.array([399, 399])

    envmap = loadtxt('maps/map4.txt')
    robotstart = np.array([0, 0])
    targetstart = np.array([5, 6])

    # envmap = loadtxt('maps/map5.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 59])
    
    # envmap = loadtxt('maps/map6.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 36])

    envmap = loadtxt('maps/map1.txt')
    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])

    env = Env(targetstart,envmap)
    t0 = tic()
    path,graph = A_star.Go_Astar(env,robotstart,epsilon=3)
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