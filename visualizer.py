
import matplotlib.pyplot as plt
from IPython import display
from collections import deque

plt.ion()

class Data_Visualizer():
    def __init__(self, max_plot_len=70):
        # self.s_tots, self.vavg_tots = [], []
        self.s_tots, self.vavg_tots = deque(), deque()
        self.laptime_est = deque()
        self.max_plot_len = max_plot_len

    def visualize(self, s_total, v_avg, game_time):
        # append to the deque's
        self.s_tots.append(s_total)
        self.vavg_tots.append(v_avg)
        if s_total != 0:
            self.laptime_est.append( game_time / (s_total / 20) )
        else: self.laptime_est.append( 0 )

        # pop from the deque's
        if len(self.s_tots) > self.max_plot_len:
            self.s_tots.popleft()
            self.vavg_tots.popleft()
            self.laptime_est.popleft()

        # display the deque's
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        # plt.title('Training time Elapsed [s]: ' + str(round(cur_time, 2)))
        plt.xlabel('Number of Games')
        plt.ylabel('s')
        plt.plot(self.s_tots)
        plt.plot(self.vavg_tots)
        plt.plot(self.laptime_est)
        plt.ylim(ymin=0)
        plt.text(len(self.laptime_est)-1, self.laptime_est[-1], str(self.laptime_est[-1]))
        # plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.01)
