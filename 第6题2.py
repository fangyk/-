# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 19:38:47 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 21:34:03 2018

@author: lenovo
"""
#得到迷宫矩阵
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg  

maze = mpimg.imread('G:/课程作业/神经网络/maze.jpg')     #读取迷宫原始图片
maze = np.delete(maze, range(504, 449, -1), axis=0) #把下方黑色部分裁剪掉
maze_gray = 0.333*maze[:, :, 0]+0.334*maze[:, :, 1]+0.333*maze[:, :, 2]#转成灰度图
maze_hat = np.zeros((450, 800))
for i in range(450):
    for j in range(800):
        if maze_gray[i, j] < 201:
            maze_hat[i, j] = 255
        elif maze_gray[i, j] > 200:
            maze_hat[i, j] = 0     #0,1化
        
#将空白部分像素缩小
maze_matrix = np.ones((59, 105))
for i in range(28):
    for j in range(52):
        maze_matrix[2*i+1, 2*j+1] = 0
        m = 0
        for k in range(15):
            m = m + maze_hat[int(15.5*i+9+k), int(15.3*j+8)]+maze_hat[int(15.5*i+9+k), int(15.3*j+9)] + \
                maze_hat[int(15.5*i+9+k), int(15.3*j+10)]
        if m < 256:
            maze_matrix[2*i+2, 2*j+1] = 0

for i in range(29):
    for j in range(51):
        maze_matrix[2 * i + 1, 2 * j + 1] = 0
        n = 0
        for k in range(15):
            n = n + maze_hat[int(15.5 * i + 9), int(15.3*j+8+k)] + maze_hat[int(15.5 * i + 10), int(15.3*j+8+k)] + \
                maze_hat[int(15.5 * i + 11), int(15.3*j+8+k)]
        if n < 256:
            maze_matrix[2 * i + 1, 2 * j + 2] = 0
maze_matrix[57, 103] = 0 #防止点跑出去
'''
#显示maze_matrix和原来图片进行比较一下
plt.imshow(maze_matrix, cmap='gray_r')
plt.axis('off')
plt.show()
'''     
#走迷宫
# 1.环境

import sys
import time
import numpy as np
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Maze(tk.Tk, object):
    UNIT = 10  # pixels
    MAZE_H = 59  # grid height
    MAZE_W = 105 # grid width

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['U', 'D', 'L', 'R']
        self.n_actions = len(self.action_space)
        self.title('迷宫')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT, self.MAZE_W * self.UNIT))       #窗口大小
        self._build_maze()

    #画矩形
    #x y 格坐标
    #color 颜色
    def _draw_rect(self, x, y, color): 
        center = self.UNIT/2
        w=center - 0
        x_=self.UNIT *x +center
        y_=self.UNIT *y +center
        return self.canvas.create_rectangle(x_-w, y_-w, x_+w, y_+w, fill = color)
    #初始化迷宫
    def _build_maze(self):
        h = self.MAZE_H*self.UNIT
        w = self.MAZE_W*self.UNIT

        #初始化画布
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
 
        # 陷阱
        self.hells=[]
        x,z=np.shape(maze_matrix)
        for i in range(0,x) :
            for j in range(0,z) :
                if maze_matrix[i,j]==1:
                    self.hells.append(self._draw_rect(j, i, 'black'))

                

        self.hell_coords = []
        for hell in self.hells:
            self.hell_coords.append(self.canvas.coords(hell)) 

        # 奖励
        self.oval = self._draw_rect(1, 1, 'yellow')

        # 玩家对象
        self.rect = self._draw_rect(103,57, 'red')

        self.canvas.pack()      #执行画 

    #重新初始化
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.rect = self._draw_rect(103,57, 'red')
        self.old_s = None
        #返回 玩家矩形的坐标 
        return self.canvas.coords(self.rect) 

    #走下一步
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT 

        #根据策略移动红块
        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect) 

        #判断是否得到奖励或惩罚
        done = False
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in self.hell_coords:
            reward = -1
            done = True
        #elif base_action.sum() == 0:
        #    reward = -1
        else:
            reward = 0
        self.old_s = s
        return s_, reward, done 

    def render(self):
        time.sleep(0.01)
        self.update()
        
class q_learning_model_maze:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.99):
        
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.q_table = pd.DataFrame(columns=actions,dtype=np.float32)
        
    #检查状态是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    #选择动作
    def choose_action(self, s):
        self.check_state_exist(s)
        if np.random.uniform() < self.e_greedy:

            state_action = self.q_table.ix[s, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  #防止相同列值时取第一个列，所以打乱列的顺序
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action
    
    #更新q表
    def rl(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]       #q估计
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.ix[s_, :].max()     #q现实
        else:
            q_target = r

        self.q_table.ix[s, a] += self.learning_rate * (q_target - q_predict)
        
               
def update():
    for episode in range(100000):
        s = env.reset()
        while True:
            env.render()
            #选择一个动作
            action = RL.choose_action(str(s))
            #执行这个动作得到反馈（下一个状态s 奖励r 是否结束done）
            s_, r, done = env.step(action)
            #更新状态表
            RL.rl(str(s), action, r, str(s_))
            s = s_
            if done:
                print(episode)
                break

 
if __name__ == "__main__":
    env = Maze()
    RL = q_learning_model_maze(actions=list(range(env.n_actions)))
    env.after(10, update)  #延迟10毫秒执行update
    env.mainloop()