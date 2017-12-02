from gym_torcs_dqn import TorcsEnv
import numpy as np
import random
import matplotlib.pyplot as plt
import math

import pickle
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from dqn import DeepQNetwork


OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 16384  #of sensors input

    np.random.seed(61502)

    vision = True

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 40000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    esar2 = []
    esar4 = []



    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)



    #from keras import backend as K
    #K.set_session(sess)



    # 1. CREATE DQN NETWORK.
    num_actions_steering = 13 # before it was 13
    num_actions_acceleration = 9 # before it was 3
    num_actions_break = 9 # before it was 3
    num_dqn_actions = num_actions_steering * num_actions_acceleration * num_actions_break
    base_dir = "/home/sergio/Projects/apclypsr/DDPG-Keras-Torcs/"
    args = {
        "save_model_freq": 1000,
        "target_model_update_freq": 1000,
        "normalize_weights": True,
        #"learning_rate": 0.00025,
        'learning_rate': 0.00025,
        "model": None
    }
    dqn = DeepQNetwork(sess, num_dqn_actions, base_dir, args)

    # Tensorflow saver
    saver = tf.train.Saver()

    #actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    #critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer



    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    # print("Now we load the weight")
    # try:
    #     # actor.model.load_weights("actormodel2.h5")
    #     # critic.model.load_weights("criticmodel2.h5")
    #     # actor.target_model.load_weights("actormodel2.h5")
    #     # critic.target_model.load_weights("criticmodel2.h5")
    #     # print("Weight load successfully")
    #      saver.restore(sess, base_dir + "dqn.ckpt")
    #      print("model restored")
    # except:
    #    print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 500) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 500 episode because of the memory leak error
        else:
            ob = env.reset()


        # 0. BUILD THE 4 images.
        s_t = np.hstack((ob.img))
        s_t_four_images_list = []
        for j in range(4):
            s_t_four_images_list.append(np.zeros((128, 128), dtype=np.float64))
        s_t_phi = get_phi_from_four_images(s_t_four_images_list)


     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])


            noise_t = np.zeros([1,action_dim])

            # 2 EVALUATE the first image
            a_t_original_dqn_discrete = dqn.inference(s_t_phi)
            #a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            # 2.5 TRANSFORM from discrete to continuous.
            a_t_original_dqn = from_discrete_actions_to_continuous_actions(a_t_original_dqn_discrete, num_actions_steering,
                                                                           num_actions_acceleration, num_actions_break)
            print("actions: ", a_t_original_dqn)

            # a_t_original[0][0] steering: [-1, 1]
            # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original_dqn[0], 0.0, 0.60, 0.30)
            # a_t_original[0][1] acceleration: [0, 1]. discretize in 6.
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original_dqn[1], 0.5, 1.00, 0.10)
            # a_t_original[0][2] break: [0, 1]
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original_dqn[2], -0.1, 1.00, 0.05)

            #The following code do the stochastic brake
            if random.random() <= 0.05:
               print("********Now we apply the brake***********")
               #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
               noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original_dqn[2], 0.2, 1.00, 0.10)

            # a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            # a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][0] = a_t_original_dqn[0] + noise_t[0][0]
            a_t[0][1] = a_t_original_dqn[1] + noise_t[0][1]
            a_t[0][2] = a_t_original_dqn[2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            # 0. UPDATE THE LAST FOUR IMAGES
            s_t1 = np.hstack((ob.img))
            if len(s_t_four_images_list) >= 4:
                s_t_four_images_list.pop(0)
                image = np.reshape(ob.img, (128, 128))
                s_t_four_images_list.append(image)

                # print greyscale image
                # plt.imshow(image, origin='lower')
                # plt.draw()
                # plt.pause(0.001)
            #get phi for the new observed state
            s_t1_phi = get_phi_from_four_images(s_t_four_images_list)

            # Add replay buffer
            #buff.add(s_t, a_t[0], r_t, s_t1, done)
            buff.add(s_t_phi,
                     from_continuous_actions_to_discrete_actions(a_t[0], num_actions_steering, num_actions_acceleration, num_actions_break),
                     r_t,
                     s_t1_phi,
                     done)  # Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            # states = np.asarray([e[0] for e in batch])
            # actions = np.asarray([e[1] for e in batch])
            # rewards = np.asarray([e[2] for e in batch])
            # new_states = np.asarray([e[3] for e in batch])
            # dones = np.asarray([e[4] for e in batch])
            # y_t = np.asarray([e[1] for e in batch])
            states = [e[0] for e in batch]
            states = np.concatenate(states, axis=0)
            actions = [e[1] for e in batch]
            rewards = [e[2] for e in batch]
            new_states = [e[3] for e in batch]
            new_states = np.concatenate(new_states, axis=0)
            dones = [e[4] for e in batch]
            y_t = [e[1] for e in batch]

            # 3. TRAINING
            #target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            # for k in range(len(batch)):
            #     if dones[k]:
            #         y_t[k] = rewards[k]
            #     else:
            #         y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                # 4 TRAIN
                loss = dqn.train(s_t = states, s_t1 = new_states, rewards = rewards, actions = actions, terminals = dones, stepNumber = step)

                # loss += critic.model.train_on_batch([states,actions], y_t)
                # a_for_grad = actor.model.predict(states)
                # grads = critic.gradients(states, a_for_grad)
                # actor.train(states, grads)
                # actor.target_train()
                # critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            esar = (i, step, a_t, r_t, loss)
            esar2.append(esar)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                # print("Now we save model")
                # actor.model.save_weights("actormodelIMG.h5", overwrite=True)
                # with open("actormodel.json", "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)
                #
                # critic.model.save_weights("criticmodelIMG.h5", overwrite=True)
                # with open("criticmodel.json", "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)
                save_path = saver.save(sess, base_dir + "dqn.ckpt")
                print("Model saved in file: %s" % save_path)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

        esar3 = (i, step, total_reward)
        esar4.append(esar3)

        def save_object(obj, filename):
            with open(filename, 'w+b') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        save_object(esar2, 'IntraEpisode.pkl')
        save_object(esar4, 'InterEpisode.pkl')

    env.end()  # This is for shutting down TORCS
    print("Finish.")
    print("Saving esars.")


def get_phi_from_four_images(s_t_four_images_list):
    s_t_four_images = np.stack(s_t_four_images_list)
    s_t_four_images = np.transpose(s_t_four_images, (1, 2, 0))
    s_t_four_images = np.reshape(s_t_four_images, (1, 128, 128, 4))
    return s_t_four_images

def from_discrete_actions_to_continuous_actions(action, num_actions_steering, num_actions_acceleration, num_actions_break):
    total_actions = num_actions_steering * num_actions_acceleration * num_actions_break
    steering_range = get_range(-1, 1, num_actions_steering)
    acceleration_range = get_range(0, 1, num_actions_acceleration)
    brake_range = get_range(0, 1, num_actions_break)
    # Convert from action index to continuous actions:
    steering_index = action % num_actions_steering
    rest = math.floor(action / num_actions_steering)
    acceleration_index = rest % num_actions_acceleration
    brake_index = math.floor(rest / num_actions_acceleration)
    return steering_range[steering_index], acceleration_range[acceleration_index], brake_range[brake_index]

def from_continuous_actions_to_discrete_actions(actions, num_actions_steering, num_actions_acceleration, num_actions_break):
    steering_c = actions[0]
    acceleration_c = actions[1]
    brake_c = actions[2]
    total_actions = num_actions_steering * num_actions_acceleration * num_actions_break
    steering_range = get_range(-1, 1, num_actions_steering)
    acceleration_range = get_range(0, 1, num_actions_acceleration)
    brake_range = get_range(0, 1, num_actions_break)
    # convert steering from continuous to discrete:
    steering_index = get_closest_index(steering_range, steering_c)
    # convert acceleration from continuous to discrete:
    acceleration_index = get_closest_index(acceleration_range, acceleration_c)
    # convert brake from continuous to discrete:
    brake_index = get_closest_index(brake_range, brake_c)
    discrete = brake_index * (num_actions_steering * num_actions_acceleration)
    discrete += acceleration_index * num_actions_steering
    discrete += steering_index
    return discrete

def get_range(lower_limit, upper_limit, num_actions):
    ranges = []
    step = (upper_limit - lower_limit) / (num_actions - 1)
    for i in range(num_actions):
        ranges.append(lower_limit + (i * step))
    return ranges

def get_closest_index(v_range, v_continuous):
    min_distance = None
    min_distance_index = 0
    for i in range(len(v_range)):
        distance = abs(v_range[i] - v_continuous)
        if min_distance is None or min_distance > distance:
            min_distance = distance
            min_distance_index = i
    return min_distance_index

if __name__ == "__main__":
    playGame()