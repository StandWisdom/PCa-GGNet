import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy

import pandas as pd
import os

from sklearn.metrics import accuracy_score
# In[Set Initial Gpu]
#ngpu= torch.cuda.device_count()
## Decide which device we want to run on
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#
#print(device)
#print(torch.cuda.get_device_name(0))

# In[]
# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

NUM_ACTIONS = 7 #0 +1 -1 +2 -2 +3 -3
NUM_STATES = 4320
ENV_A_SHAPE = 0

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)       

    def forward(self,x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)       
        x = F.relu(x)
        
        x = self.out(x)
#        x = F.relu(x)
        x = torch.sigmoid(x)
        x = nn.Softmax(dim=1)(x)      
        return x
    
class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        if self.memory_counter % 500 ==0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
#        print(batch_state.size)
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return q_eval,q_next

class pca_env():
    def __init__(self, rewardListPath=None,gameRate=1):
        self.df = pd.read_csv(rewardListPath)
        self.gameList = list(set(list(self.df['ID'])))  
        self.gameList = self.gameList[:int(gameRate*len(self.gameList))]   
        self.labelpath = './data/ID-lebels.csv'
        self.df_label = pd.read_csv(self.labelpath)
        
        self.gameId = 0
        print('Init pca_env Class.')
    
    def creat_game(self,shuffle=True):
        if shuffle:
            train_id = np.random.randint(0,len(self.gameList))
        else:
            train_id = self.gameId
            self.gameId += 1
            
        game_name = self.gameList[train_id]
        self.df_game = self.df[self.df['ID']==game_name]
        self.goal = list(set(list(self.df_game[self.df_game['label']!=5]['label'])))[0]
        self.cnnPred = list(self.df_game['pred'])
#        self.cnnScore = np.array(np.max(self.df_game[['ISUP0','ISUP1','ISUP2','ISUP3','ISUP4']],axis=1)) # OLD
        self.cnnScore = np.array(self.df_game[['ISUP0','ISUP1','ISUP2','ISUP3','ISUP4']])[:,int(self.goal)]

        self.game = np.array(self.df_game.iloc[:,9:]).tolist()
        print('Current game:{},Goal:{}.'.format(game_name,self.goal))
        return self.game,self.goal,self.cnnPred,self.cnnScore
        
    def reset(self,resetType=0): # 0:random; 1:median; 2:first slice; 3: random median
        if resetType==0:
            self.idx = np.random.randint(0,len(self.game))
        elif resetType==1:
            self.idx = len(self.game)//2
        elif resetType==2:
            self.idx = 0
        elif resetType==3:
            self.idx = len(self.game)//2 + np.random.randint(-5,5)
        else:
            print('Unknow reset type. Please check the resetType')
            
        state = self.game[self.idx]
        return np.array(state)
        
    def step(self,action):
        old_index = self.idx
        if action==0:
            self.idx = self.idx
        elif action==1:
            self.idx = self.idx-1
        elif action==2:
            self.idx = self.idx+1 
        elif action==3:
            self.idx = self.idx-2 
        elif action==4:
            self.idx = self.idx+2 
        elif action==5:
            self.idx = self.idx-3  
        elif action==6:
            self.idx = self.idx+3            
        
        if self.idx<0 or self.idx>len(self.game)-1:
            done = True
            new_index = old_index
        else:
            done = False
            new_index = self.idx
            
        state = self.game[new_index]
        pred = self.cnnPred[new_index]
        score = self.cnnScore[new_index]
        
#        print(list(self.df_game['label']))
        if list(self.df_game['label'])[new_index]==5:
            tumorFlag = False
        else:
            tumorFlag = True
        
        if tumorFlag and self.goal==pred:
            done = True
        
        return state,done,pred,score,tumorFlag

    
def reward_func(goal, pred, score, tumorFlag):
    alpha = 0.5
    if tumorFlag and pred==goal:
        reward = 1+ alpha*score   
    elif tumorFlag:
        reward = 1
    else:  
        reward = 0
         
    return reward

# In[]  

TrainFlag = 0  

Dir = './models2/pnasnet5large'
gameDir = 'Path of your gamefile' # Your gamefiles
gameFile = gameDir+'-ft.csv'

cnnModelDir = os.path.join(Dir,gameDir)
itera = 7000
MODELPATH = os.path.join(cnnModelDir,gameDir+'-'+str(itera)+'-DQN.pt')

whileCounts = 0 # Only for train
if __name__ == '__main__':
#    print('**')
    print('Creat Net')
    dqn = DQN()
    print('Creat Env')
    env = pca_env(os.path.join(Dir,gameDir,gameFile),gameRate=1)
    
    # Train
    if TrainFlag:
        episodes = 10001 # 
        reward_list = []
        for game_i in range(0,episodes): 
            game,goal,pred,score = env.creat_game(shuffle=True)
            print(game_i,"Collecting Experience....")
            state = env.reset(resetType=3) # # 0:random; 1:median; 2:first slice; 3: random median
            ep_reward = 0
            while True:
                action = dqn.choose_action(state)
                next_state, done, pred, score, tumorFlag = env.step(action)    
    
                reward = reward_func(goal, pred, score, tumorFlag)        
                dqn.store_transition(state, action, reward, next_state)
                ep_reward += reward
    #            print(action,goal,pred, score)
                if dqn.memory_counter >= MEMORY_CAPACITY:
                    q_eval,q_next = dqn.learn()
                    if done:
                        print("Game: {} , the episode reward is {}".format(game_i, ep_reward))
                        print('Action:{},goal:{},pred:{},scores:{}'.format(action,goal,pred, score))
                if done or whileCounts>9999:
                    whileCounts=0
                    break
                state = next_state
                whileCounts +=1
                
            r = copy.copy(reward)
            reward_list.append(r)
            
            if game_i!=0 and game_i % 1000==0:
                torch.save({'model_state_dict': dqn.target_net.state_dict(),
                            'optimizer_state_dict': dqn.optimizer.state_dict(),
                            'epochs':episodes},
                            os.path.join(cnnModelDir,gameDir+'-'+str(game_i)+'-DQN.pt'))    
                
    if not TrainFlag:
        print('Load model')
        model = Net()
        model.load_state_dict(torch.load(MODELPATH)['model_state_dict'])
        model.eval() 
        
        final_Pred = []
        
        episodes= len(env.gameList) # Total env num
        episodes= 200

        action_all_L=[]
        for game_i in range(0,episodes):
    #    for game_i in range(0,10):
            print('---------------------')
            game,goal,pred,score = env.creat_game(shuffle=False)
            state = env.reset(resetType=1) # 0:random; 1:median; 2:first slice; 3: random median
            
            circle = 4
            action_flag = 1
            action_L=[]
            action_L.append(env.gameId)
            while circle>0:
                circle -=1
                # Choose actions
                state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
                action_value = model.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()[0]
                # Update step                  
                next_state, done, pred, score, tumorFlag = env.step(action)  
                
                action_L.append([env.idx,goal,action,done,pred,score,tumorFlag])# record action
                
                if action ==0:
                    action_flag -=1
                    break
                
                state = next_state
            action_all_L.append(action_L)
            print('Action num:{},goal:{},pred:{},scores:{}'.format(circle,goal,pred, score))
            final_Pred.append([env.gameList[game_i],circle,tumorFlag,goal,pred,score])
        
        df_save = pd.DataFrame(final_Pred,columns=['ID','rounds','tumorFlag','goal','pred','score'])
            
            
            
            
            
            
            
            
            
        
        
        
        
    