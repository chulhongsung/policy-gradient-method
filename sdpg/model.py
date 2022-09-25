import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt

from collections import deque
import random
import gym 

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## 버퍼에 저장
    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: # 찼으면 가장 오래된 데이터 삭제하고 저장
            self.buffer.popleft()
            self.buffer.append(transition)

    ## 버퍼에서 데이터 무작위로 추출 (배치 샘플링)
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # 상태, 행동, 보상, 다음 상태별로 정리
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones


    ## 버퍼 사이즈 계산
    def buffer_count(self):
        return self.count


    ## 버퍼 비움
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0

class Critic(K.models.Model):
    """
    Critic network which returns parameters of the linear isotonic regression spline 

    Args:
        hidden_dims (list): A list consists of hidden layers output dimensions 
        param_dims (int): An integer value denotes the number of spline parameters such as knots(delta), slope(beta), ...
        min_reward (int): min_reward <= reward <= max_reward, if min_reward is not None, then parameter gamma = min_reward   
        inputs (tensor): tf.concat([temp_state, temp_action], axis=-1) 
    """
    
    def __init__(self):
        super(Critic, self).__init__()   
        self.h_s = [K.layers.Dense(dim, activation='relu',kernel_initializer=K.initializers.GlorotUniform()) for dim in [400, 300]] 
        self.h_a = [K.layers.Dense(dim, activation='relu',kernel_initializer=K.initializers.GlorotUniform()) for dim in [400, 300]] 
        self.h_n = K.layers.Dense(300, activation='relu',kernel_initializer=K.initializers.GlorotUniform())
        self.value_map = K.layers.Dense(1, kernel_initializer=K.initializers.GlorotUniform())                     
        
    @tf.function
    def call(self, input):
        state, action, noise = input
        h_state = self.h_s[1](self.h_s[0](state))
        h_action = self.h_a[1](self.h_a[0](action))
        h_noise = self.h_n(noise)
        
        value = self.value_map(tf.expand_dims(h_state, 1) + tf.expand_dims(h_action, 1) + h_noise)        
        return value

class Actor(K.models.Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = K.layers.Dense(400, activation='relu',kernel_initializer=K.initializers.GlorotUniform())
        self.h2 = K.layers.Dense(300, activation='relu', kernel_initializer=K.initializers.GlorotUniform())
        self.action = K.layers.Dense(action_dim, activation='tanh', kernel_initializer=K.initializers.GlorotUniform())

    @tf.function 
    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        a = self.action(x)

        a = K.layers.Lambda(lambda x: x*self.action_bound)(a)

        return a

class Critic(K.models.Model):
    """
    Critic network which returns parameters of the linear isotonic regression spline 

    Args:
        hidden_dims (list): A list consists of hidden layers output dimensions 
        param_dims (int): An integer value denotes the number of spline parameters such as knots(delta), slope(beta), ...
        min_reward (int): min_reward <= reward <= max_reward, if min_reward is not None, then parameter gamma = min_reward   
        inputs (tensor): tf.concat([temp_state, temp_action], axis=-1) 
    """
    
    def __init__(self):
        super(Critic, self).__init__()   
        self.h_s = [K.layers.Dense(dim, activation='relu',kernel_initializer=K.initializers.GlorotUniform()) for dim in [400, 300]] 
        self.h_a = [K.layers.Dense(dim, activation='relu',kernel_initializer=K.initializers.GlorotUniform()) for dim in [400, 300]] 
        self.h_n = K.layers.Dense(300, activation='relu',kernel_initializer=K.initializers.GlorotUniform())
        self.value_map = K.layers.Dense(1, kernel_initializer=K.initializers.GlorotUniform())                     
        
    @tf.function
    def call(self, input):
        state, action, noise = input
        h_state = self.h_s[1](self.h_s[0](state))
        h_action = self.h_a[1](self.h_a[0](action))
        h_noise = self.h_n(noise)
        
        value = self.value_map(tf.expand_dims(h_state, 1) + tf.expand_dims(h_action, 1) + h_noise)        
        return value

class QuantileHuber(K.losses.Loss):
    def __init__(self, num_atoms=51, delta=1):
        super(QuantileHuber, self).__init__()
        self.num_atoms = num_atoms # num class
        self.delta = delta # image row * col

        min_tau = 1/(2*self.num_atoms)
        max_tau = (2*self.num_atoms+1)/(2*self.num_atoms)
        self.tau = tf.reshape (tf.range(min_tau, max_tau, 1/self.num_atoms), [1, self.num_atoms])
        self.inv_tau = 1.0 - self.tau 
        
    def call(self, pred, target):
        error = pred - tf.transpose(target, perm=[0, 2, 1])
        tmp_loss = tf.where(tf.less(tf.abs(error), 1.0), 0.5*tf.math.square(error), tf.abs(error) - 0.5 )
        loss = tf.where(tf.less(error, 0.0), self.inv_tau * tmp_loss, self.tau * tmp_loss)
        
        return tf.math.reduce_mean(loss)

class SDPGagent(object):
    
    def __init__(self, env, seed):
        self.GAMMA = 0.9
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 1000000
        self.ACTOR_LEARNING_RATE = 0.00003
        self.CRITIC_LEARNING_RATE = 0.00003
        self.SAMPLE_SIZE = 51
        self.DELTA = 0.3
        self.DELTA_DECAYING_RATE = 0.9999
        self.TAU = 0.01
        self.SEED = seed
        
        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        
        self.env.seed(self.SEED)
        tf.random.set_seed(self.SEED)
        
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)
        
        self.critic = Critic()
        self.target_critic = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
                
        actor_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.ACTOR_LEARNING_RATE, 10000, 0.999, staircase=False, name=None
            )

        critic_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.CRITIC_LEARNING_RATE, 10000, 0.999, staircase=False, name=None
            )

        self.optimizer_theta = K.optimizers.Adam(learning_rate=actor_lr_schedule)
        self.optimizer_phi = K.optimizers.Adam(learning_rate=critic_lr_schedule)
    
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        
        self.qh_loss = QuantileHuber()
        
        self.save_epi_reward = []
    
    def gaussian_noise(self, mu=0.0, sigma=1):
        return tf.random.normal([self.action_dim], mu, sigma, tf.float32)
    
    def sample_noise(self, mu=0.0, sigma=1):
        return tf.random.normal([self.BATCH_SIZE, self.SAMPLE_SIZE, 1], mu, sigma, tf.float32)

    def update_target_network(self, TAU):
        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)
        
    
    @tf.function
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            noise = self.sample_noise()
            z = self.critic([states, self.actor(states), noise])
            mean_value = tf.math.reduce_mean(z) 
    
        grad_theta = tape.gradient(mean_value, self.actor.weights)
        self.optimizer_theta.apply_gradients(zip(grad_theta, self.actor.weights))
        
    @tf.function 
    def critic_learn(self, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            noise1 = self.sample_noise()
            noise2 = self.sample_noise()            
            z = self.critic([states, actions, noise1])
            next_action = self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
            z_tilde = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32), next_action, noise2])
            z_tilde = tf.sort(z_tilde, axis=1)
            rewards_ = tf.reshape(rewards, [self.BATCH_SIZE, 1, 1])
            rewards = tf.cast(rewards_, dtype=tf.float32)
            pred = rewards + self.GAMMA * z_tilde 
            target = tf.sort(z, axis=1)
            loss = self.qh_loss(pred, target)

        grad_phi = tape.gradient(loss, self.critic.weights)
        self.optimizer_phi.apply_gradients(zip(grad_phi, self.critic.weights))
    
    def load_weights(self, path):
        self.actor.load_weights(path + 'actor.h5')
        self.critic.load_weights(path + 'critic.h5')
        
    def train(self, MAX_EPISODE_NUM):
        
        random.seed(self.SEED)
        tf.random.set_seed(self.SEED)
        
        self.update_target_network(1.0)
        
        for ep in range(MAX_EPISODE_NUM):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()
            
            while not done:
                
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.gaussian_noise()
                # 행동 범위 클리핑
                action = np.clip(action + self.DELTA * (self.DELTA_DECAYING_RATE)**ep * noise, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # 학습용 보상 설정
                train_reward = reward * 1.0
                # 리플레이 버퍼에 저장
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:
                    
                    states, actions, rewards, next_states, _ = self.buffer.sample_batch(self.BATCH_SIZE)
                    states = tf.cast(states, dtype=tf.float32)
                    next_states = tf.cast(next_states, dtype=tf.float32)
                    # dones = tf.cast(dones, tf.float32)

                    self.critic_learn(states, actions, rewards, next_states)
                    self.actor_learn(states)
                
                    self.update_target_network(self.TAU)
                    
                state = next_state
                episode_reward += reward
                time += 1
                
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
                
            self.save_epi_reward.append(round(episode_reward, 2))
            
    def plot_result(self):
        plt.style.use(['default'])
        plt.plot(self.save_epi_reward)
        plt.show()
            
env = gym.make("BipedalWalker-v3")
agent = SDPGagent(env, seed=10)
MAX_EPISODE_NUM = 10000
agent.train(MAX_EPISODE_NUM)
agent.plot_result()
