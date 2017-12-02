import pickle as pc
import seaborn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


interepisode = pc.load(open("/home/sergio/Projects/apclypsr/DDPG-Keras-Torcs/ddpg_baseline_1175/InterEpisode.pkl", "rb"))

print(len(interepisode))

rewards = []
j = 0

for i in range(len(interepisode)):
    j += 1
    if j > 2:
        rewards.append(interepisode[i][2] / (interepisode[i][1] - interepisode[i - 1][1]))

plt.subplot(1, 1, 1)
label_steps, = plt.plot(np.array(rewards), label='normalized_rewards')
plt.legend(handles=[label_steps])
plt.draw()
plt.show()