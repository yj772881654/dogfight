import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch
from torch.autograd import Variable
import random
import os
import numpy as np
from collections import deque
from math import cos,sin,sqrt,pow,acos,pi
import torch.nn.functional as F
ACTIONS = 7 # number of valid actions
GAMMA = 0.9 # decay rate of past observations
OBSERVE = 50. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 50 # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 10

class FCN(nn.Module):
    def __init__(self, n_states=18, n_actions=7):
        """ 初始化q网络，为全连接网络
            n_states: 输入的feature即环境的state数目
            n_actions: 输出的action总个数
        """
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)  # 输入层
        self.fc2 = nn.Linear(128, 128)  # 隐藏层
        self.fc3 = nn.Linear(128, n_actions)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class BrainDQNMain(object):
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    def load(self):
        if os.path.exists("params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    def __init__(self,actions):
        self.replayMemory = deque() # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.Q_net=FCN()
        self.Q_netT=FCN()
        self.load()
        self.loss_func=nn.MSELoss()
        LR=5e-3
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

        self.currentState=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    def train(self): # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch] # Step 2: calculate y
        # print(type(state_batch),type(action_batch),type(reward_batch),type(nextState_batch))
        y_batch = np.zeros([BATCH_SIZE,1])
        nextState_batch=np.array(nextState_batch) #print("train next state shape")
        #print(nextState_batch.shape)
        nextState_batch=torch.Tensor(nextState_batch)
        action_batch=np.array(action_batch)

        # print("action_batch:")
        # for each in action_batch:
        #     print(each)

        index=action_batch.argmax(axis=1)
        # print("action "+str(index))
        index=np.reshape(index,[BATCH_SIZE,1])
        action_batch_tensor=torch.LongTensor(index)
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch=QValue_batch.detach().numpy()

        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0]=reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch=np.array(y_batch)
        y_batch=np.reshape(y_batch,[BATCH_SIZE,1])
        state_batch_tensor=Variable(torch.Tensor(state_batch))
        y_batch_tensor=Variable(torch.Tensor(y_batch))


        # print(action_batch_tensor,action_batch.shape)
        y_predict=self.Q_net(state_batch_tensor).gather(1,action_batch_tensor)
        # print("y_predict",y_predict,"y_batch_tensor",y_batch_tensor)
        loss=self.loss_func(y_predict,y_batch_tensor)
        # print("loss is "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    def setPerception(self,nextObservation,action,reward,terminal): #print(nextObservation.shape)
        newState = np.array(nextObservation)
        action_onehot = np.zeros(self.actions)
        # print(action_onehot.shape)
        action_onehot[action]=1
        self.replayMemory.append((self.currentState,action_onehot,reward,newState,terminal))
        # print(self.currentState.shape,action.shape,type(reward),newState.shape)
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE: # Train the network
            self.train()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1
    def getAction(self):
        currentState = torch.from_numpy(self.currentState).reshape(1,-1)
        currentState=currentState.float()

        QValue = self.Q_netT(currentState)[0]
        # print(currentState)
        # print(self.Q_net(currentState))
        # print(self.Q_netT(currentState))
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue.detach().numpy())
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action