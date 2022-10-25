import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import loadtxt
from pqdict import pqdict
'''
The simulation of AMRA

'''

def tic():
      return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

class Env:
  def __init__(self,target_pos,envmap):
    self.goal_node = AMRA_star.node(None,tuple(target_pos))
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

class AMRA_star:
  '''
  def the class for A star algorithm
  '''
  def __init__(self) -> None:
    
    self.open_list = [pqdict({}), pqdict({})]
    self.envmap = envmap

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
    

  def motion_model(i):
    '''
    The motion model for resolution =1 map, need change for multi-resolution
    
    '''
    if (i==0):
        motion = [[-1, 0], # move up
                  [0 ,-1], # move left
                  [1 , 0], # move down
                  [0 , 1], # move right
                  [-1, 1], # move up-right
                  [-1,-1], # move up-left
                  [1 ,-1], # move down-left
                  [1 , 1]] # move down-right
    elif (i==1):
        motion = [[-2, 0], # move up
                  [0 ,-2], # move left
                  [2 , 0], # move down
                  [0 , 2], # move right
                  [-2, 2], # move up-right
                  [-2,-2], # move up-left
                  [2 ,-2], # move down-left
                  [2 , 2]] # move down-right
                  

    return motion

  def action_back(child_node,parent_node):
      if parent_node is not None:
        x_move = child_node.pos[0] - parent_node.pos[0]
        y_move = child_node.pos[1] - parent_node.pos[1]
        move = (x_move,y_move)
        motion = AMRA_star.motion_model(0)
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
      action.insert(0,AMRA_star.action_back(current_node,optimal_parent))
      current_node = optimal_parent

    return action

  def Key(x, env, epsilon):
        #TODO add the cost to come value
    x.g + self.w1 * env.getHeuristic(x, epsilon)

  def Go_AMRAstar(self,env,start_pos,epsilon=1):
    self.w1 = 6
    self.w2 = 6

    bp = []

    start_node = AMRA_star.node(None,tuple(start_pos))
    start_node.g = 0
    start_node.h = env.getHeuristic(start_node,epsilon)
    start_node.f = start_node.g + start_node.h

    Incons = [start_node]

    while self.w1 >= 1 and self.w2 >= 1:

        for x in Incons:
            self.open_list[0][x] = self.Key(x, env, epsilon)

        Incons.clear

        envmap = env.map
        state_graph = {}
        len_row, len_col = env.map.shape
        for idx in range(len_row):
          for jdx in range(len_col):
            key = str((idx,jdx))
            state_graph[key] = None

        state_graph[str(tuple(start_pos))] = {'open?':True,'node':start_node}

        env.goal_node.g = env.goal_node.f = np.inf
        env.goal_node.h = env.getHeuristic(env.goal_node,epsilon)

        for x in self.open_list[0]:
            for j in range(self.open_list.len()):
                self.open_list[j][str(tuple(x.pos))] = self.Key(x, env, epsilon)
                closed_list = [[],[]]

        iteration = 0
        len_row, len_col = env.map.shape
        motion = AMRA_star.motion_model(iteration % 2)
        print('A-star calculating...')

        while len(self.open_list) > 0:
            # while end_node not in closed_list:
            iteration+=1  

            motion = AMRA_star.motion_model(iteration % 2)
        
            smallest_node_pos = self.open_list[iteration % 2].popitem()[0]
            # print('iteration in: '+ str(iteration) + smallest_node_pos )

            closed_list[iteration % 2].append( state_graph[smallest_node_pos]['node'])

            state_graph[smallest_node_pos]['open?'] = False

            if  state_graph[smallest_node_pos]['node'] ==  env.goal_node:
                print("A-star return!")
                    # return open_node
                return AMRA_star.recover_path(state_graph[smallest_node_pos]['node']), state_graph
                # return state_graph

            stage_cost = 1 
            for closed_node in closed_list[iteration % 2]:
             for move in motion:
                # print('one move')
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
                    child  = AMRA_star.node(closed_node,tuple(node_pos))
                    child.g = closed_node.g + stage_cost
                    child.h = env.getHeuristic(child,epsilon)
                    child.f = child.g + child.h

                    state_graph[str(node_pos)] = {'open?':True,'node':child}

                    self.open_list[str(node_pos)] = child.f

                else:
                    state_graph[str(node_pos)]['node'].parent.append(closed_node)
                    state_graph[str(node_pos)]['node'].g = min((closed_node.g + stage_cost),state_graph[str(node_pos)]['node'].g)
                    state_graph[str(node_pos)]['node'].f = state_graph[str(node_pos)]['node'].g + state_graph[str(node_pos)]['node'].h
                    state_graph[str(node_pos)]['open?'] = True

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

    envmap = loadtxt('maps/map1.txt')
    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])

    # envmap = loadtxt('maps/map7.txt')
    # robotstart = np.array([1, 1])
    # targetstart = np.array([4998, 4998])

    env = Env(targetstart,envmap)
    t0 = tic()
    
    Astar = AMRA_star()

    path,graph = Astar.Go_AMRAstar(env,robotstart,epsilon=3)


    
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
