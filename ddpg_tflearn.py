from gym_torcs_dqn import TorcsEnv
import numpy as np
import random

import pickle
import argparse
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork_tflearn import ActorNetwork
from CriticNetwork_tflearn import CriticNetwork
from OU import OU
import timeit

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    np.random.seed(61502)

    base_dir = "/home/sergio/Projects/apclypsr/DDPG-Keras-Torcs/"

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 1800
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    esar2 = []
    esar4 = []

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    #tf.set_random_seed(61502)

    actor = ActorNetwork(sess, state_dim, action_dim,
                         LRA, TAU, BATCH_SIZE)

    critic = CriticNetwork(sess, state_dim, action_dim,
                           LRC, TAU, GAMMA,
                           actor.get_num_trainable_vars())





    #actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    #critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Now load the weight

    restore = False
    if restore:
        print("Now we load the weight")
        # tf.reset_default_graph()

        # Tensorflow saver
        saver = tf.train.Saver()
        try:
            saver.restore(sess, base_dir + "ddpg.ckpt")
            print("model restored")
        except:
            print("Cannot find the weight")
    else:
        print("No weight loaded")
        init = tf.global_variables_initializer()
        sess.run(init)
        # Tensorflow saver
        saver = tf.train.Saver()

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 500) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 500 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        ep_ave_max_q = 0
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])

            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.predict(s_t.reshape(1, s_t.shape[0]))
            # print("a_t_original")
            print(a_t_original)
            # print(a_t_original.shape)
            # print(a_t_original[0,1])
            # print(a_t_original[0][1])

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            # The following code do the stochastic brake
            # if random.random() <= 0.05:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            if buff.size() > BATCH_SIZE:
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])

                actor_predicted_actions = actor.predict_target(new_states)
                #print("Actor predicted actions: ", actor_predicted_actions.shape)
                #print("New states: ", new_states.shape)

                target_q_values = critic.predict_target(new_states, actor_predicted_actions)

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                if (train_indicator):
                    # loss += critic.model.train_on_batch([states, actions], y_t)
                    # a_for_grad = actor.model.predict(states)
                    # grads = critic.gradients(states, a_for_grad)
                    # actor.train(states, grads)
                    # actor.target_train()
                    # critic.target_train()

                    # Update the critic given the targets

                    print("y_t")
                    print(y_t.shape)

                    predicted_q_value, _, loss, loss2 = critic.train(states, actions, y_t)

                    print("LOSS:", loss)
                    print("LOSS2:", loss2)

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(states)
                    grads = critic.action_gradients(states, a_outs)
                    actor.train(states, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                #batch update

            #step end
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
                # actor.model.save_weights("actormodel2.h5", overwrite=True)
                # with open("actormodel.json", "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)
                #
                # critic.model.save_weights("criticmodel2.h5", overwrite=True)
                # with open("criticmodel.json", "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)
                save_path = saver.save(sess, base_dir + "ddpg.ckpt")
                print("Model saved in file: %s" % save_path)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

        esar3 = (i, step, total_reward)
        esar4.append(esar3)

        def save_object(obj, filename):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        save_object(esar2, 'IntraEpisode.pkl')
        save_object(esar4, 'InterEpisode.pkl')



    env.end()  # This is for shutting down TORCS
    print("Finish.")
    print("Saving esars.")




if __name__ == "__main__":
    playGame()