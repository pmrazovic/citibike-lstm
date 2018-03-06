from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

# ---------------- MAE: LSTM for different k

time_slots = [1,2,3,4,5,6,7,8]

pick_up_lstm_4 = [0.3, 0.35, 0.37, 0.43, 0.45, 0.5, 0.58, 0.59]
pick_up_lstm_8 = [0.4, 0.45, 0.47, 0.49, 0.6, 0.61, 0.68, 0.69]
pick_up_lstm_16 = [0.6, 0.65, 0.67, 0.69, 0.71, 0.72, 0.78, 0.81]
pick_up_lstm_24 = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]

drop_offs_lstm_4 = [0.3, 0.35, 0.37, 0.43, 0.45, 0.5, 0.58, 0.59]
drop_offs_lstm_8 = [0.4, 0.45, 0.47, 0.49, 0.6, 0.61, 0.68, 0.69]
drop_offs_lstm_16 = [0.6, 0.65, 0.67, 0.69, 0.71, 0.72, 0.78, 0.81]
drop_offs_lstm_24 = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]

fig = plt.figure()

plot_1 = fig.add_subplot(2,1,1)

color = plt.cm.tab20([0,1,2,3,4,5,6])
hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),tuple(color[:,0:-1]))

plt.rcParams['axes.color_cycle'] = hexcolor

p_4, = plot_1.plot(time_slots, pick_up_lstm_4, label="k = 4", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_8, = plot_1.plot(time_slots, pick_up_lstm_8, label="k = 8", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_16, = plot_1.plot(time_slots, pick_up_lstm_16, label="k = 16", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_24, = plot_1.plot(time_slots, pick_up_lstm_24, label="k = 24", linestyle='-', linewidth=1, marker = 'o', markersize=4)
plot_1.legend(handles=[p_4,p_8,p_16,p_24])
plot_1.grid(linestyle='-', linewidth=0.3)
plot_1.set_title('(a) MAE of pick-up demand')
plot_1.set_ylabel('MAE [# pick-ups]')

plot_2 = fig.add_subplot(2,1,2)


d_4, = plot_2.plot(time_slots, drop_offs_lstm_4, label="k = 4", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_8, = plot_2.plot(time_slots, drop_offs_lstm_8, label="k = 8", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_16, = plot_2.plot(time_slots, drop_offs_lstm_16, label="k = 16", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_24, = plot_2.plot(time_slots, drop_offs_lstm_24, label="k = 24", linestyle='-', linewidth=1, marker = 'o', markersize=4)
plot_2.legend(handles=[d_4,d_8,d_16,d_24])
plot_2.grid(linestyle='-', linewidth=0.3)

plot_2.set_title('(b) MAE of drop-off demand')
plot_2.set_ylabel('MAE [# drop-offs]')
plot_2.set_xlabel('Future time interval')

fig.show()

# ---------------- MAE: LSTM vs other methods

pick_up_lstm = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
pick_up_svr = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
pick_up_mlp = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
pick_up_rt= [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
pick_up_lr = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]

drop_off_lstm = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
drop_off_svr = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
drop_off_mlp = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
drop_off_rt= [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
drop_off_lr = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]

fig = plt.figure()

color = plt.cm.tab20([0,1,2,3,4,5,6])
hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),tuple(color[:,0:-1]))

plot_1 = fig.add_subplot(2,1,1)

p_lstm, = plot_1.plot(time_slots, pick_up_lstm, label="LSTM", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_svr, = plot_1.plot(time_slots, pick_up_svr, label="SVR", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_mlp, = plot_1.plot(time_slots, pick_up_mlp, label="MLP", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_rt, = plot_1.plot(time_slots, pick_up_rt, label="RT", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_lr, = plot_1.plot(time_slots, pick_up_lr, label="LR", linestyle='-', linewidth=1, marker = 'o', markersize=4)

plot_1.legend(handles=[p_lstm,p_svr,p_mlp,p_rt,p_lr])
plot_1.grid(linestyle='-', linewidth=0.3)
plot_1.set_title('(a) MAE of pick-up demand')
plot_1.set_ylabel('MAE [# pick-ups]')
plot_1.set_xlabel('Future time interval')

plot_2 = fig.add_subplot(2,1,2)

d_lstm, = plot_2.plot(time_slots, drop_off_lstm, label="LSTM", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_svr, = plot_2.plot(time_slots, drop_off_svr, label="SVR", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_mlp, = plot_2.plot(time_slots, drop_off_mlp, label="MLP", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_rt, = plot_2.plot(time_slots, drop_off_rt, label="RT", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_lr, = plot_2.plot(time_slots, drop_off_lr, label="LR", linestyle='-', linewidth=1, marker = 'o', markersize=4)

plot_2.legend(handles=[d_lstm,d_svr,d_mlp,d_rt,d_lr])
plot_2.grid(linestyle='-', linewidth=0.3)
plot_2.set_title('(b) MAE of drop-off demand')
plot_2.set_ylabel('MAE [# drop-off]')
plot_2.set_xlabel('Future time interval')

fig.show()

# ---------------- MAE: LSTM - separate models

pick_up_collective = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
pick_up_separate = [0.6, 0.65, 0.67, 0.69, 0.71, 0.72, 0.78, 0.81]

drop_off_collective = [0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.98, 0.99]
drop_off_separate = [0.6, 0.65, 0.67, 0.69, 0.71, 0.72, 0.78, 0.81]

fig = plt.figure()

color = plt.cm.tab20([0,1,2,3,4,5,6])
hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),tuple(color[:,0:-1]))

plot_1 = fig.add_subplot(2,1,1)

p_sep, = plot_1.plot(time_slots, pick_up_separate, label="Separate", linestyle='-', linewidth=1, marker = 'o', markersize=4)
p_col, = plot_1.plot(time_slots, pick_up_collective, label="Collective", linestyle='-', linewidth=1, marker = 'o', markersize=4)

plot_1.legend(handles=[p_sep,p_col])
plot_1.grid(linestyle='-', linewidth=0.3)
plot_1.set_title('(a) MAE of pick-up demand')
plot_1.set_ylabel('MAE [# pick-ups]')

plot_2 = fig.add_subplot(2,1,2)

d_sep, = plot_2.plot(time_slots, drop_off_separate, label="Separate", linestyle='-', linewidth=1, marker = 'o', markersize=4)
d_col, = plot_2.plot(time_slots, drop_off_collective, label="Collective", linestyle='-', linewidth=1, marker = 'o', markersize=4)

plot_2.legend(handles=[d_sep,d_col])
plot_2.grid(linestyle='-', linewidth=0.3)
plot_2.set_title('(a) MAE of drop-off demand')
plot_2.set_ylabel('MAE [# drop-off]')
plot_2.set_xlabel('Future time interval')

fig.show()
















