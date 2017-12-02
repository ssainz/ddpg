import pickle as pc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


inter = pc.load(open("/home/sergio/Projects/apclypsr/DDPG-Keras-Torcs/ddpg_tflearn_1/InterEpisode.pkl", "rb"))


steps_trend = []

prev = 0
for tuple in inter:

    steps_trend.append(tuple[1] - prev)

    prev = tuple[1]


plt.subplot(1, 1, 1)
label_steps, = plt.plot(np.array(steps_trend), label='trend_steps')
plt.legend(handles=[label_steps])
plt.draw()
plt.show()

