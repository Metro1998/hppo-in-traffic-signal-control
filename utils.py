import torch
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch.nn as nn

from brokenaxes import brokenaxes


### plot
def visualize_overall_agent_results(agent_results, agent_name: list, file_path_for_pic, tile, y_label, y_min, y_max):
    """
    Visualize the results for one agent.
    :param y_label:
    :param tile:
    :param file_path_for_pic:
    :param agent_results: list of lists, each
    :param agent_name:
    :return:
    """
    # assert isinstance(agent_results, list), 'agent_results must be a list of list of list.'
    # assert isinstance(agent_results[0], list), 'agent_result must be a list of list of list.'
    # assert isinstance(agent_results[0][0], list), 'agent_result must be a list of list of list.'
    # x_major_locator = MultipleLocator(5)
    plt.rcParams['axes.unicode_minus'] = False
    agent_to_color_dictionary = {
        'continuous': '##0099DD',
        # 'discrete': '#FF69B4',
        # 'hybrid': '#800080',
        'sequence 1': '#0099DD',
        'sequence 2': '#FF69B4',
        'sequence 3': '#800080',
        'light': '#0099DD',
        'medium': '#FF69B4',
        'heavy': '#800080',
        '5 second': '#FF69B4',
        '10 second': '#0099DD',
        '15 second': '#F25A38',
        '20 second': '#800080',
        '25 second': '#032CA6',
        '30 second': '#D91E41',
        '11 second': '#FF69B4',
        '12 second': '#0099DD',
        '13 second': '#F25A38',
        '14 second': '#800080',
        '16 second': '#FF69B4',
        '17 second': '#0099DD',
        '18 second': '#F25A38',
        '19 second': '#800080',
        '21 second': '#FF69B4',
        '22 second': '#0099DD',
        '23 second': '#F25A38',
        '24 second': '#800080',
        'FRAP': '#800080',

        'discrete': '#FF69B4',
        'continuous opposite': '#0099DD',
        'hybrid': '#F25A38',
        'hyar': '#F25A38',
        'continuous single': '#800080',

        'maximum delay': '#800080',
        'average delay': '#0099DD',
        'queue length': '#FF69B4',
    }
    plt.figure(figsize=(8, 6), tight_layout=True)

    ax = plt.subplot(111)
    ax.grid(color='grey', lw=0.25)

    ax.spines['bottom'].set_linewidth('1.5')
    ax.spines['top'].set_linewidth('1.5')
    ax.spines['left'].set_linewidth('1.5')
    ax.spines['right'].set_linewidth('1.5')

    ax.set_facecolor('xkcd:white')
    ax.set_xlabel('Episode', fontdict={'font': 'Times New Roman', 'fontsize': 16}, labelpad=5)
    ax.set_ylabel('Average queue length (veh)', fontdict={'font': 'Times New Roman', 'fontsize': 16}, labelpad=5)
    # for spine in ['right', 'top']:
    # ax.spines[spine].set_visible(False)

    for i in range(4):
        color = agent_to_color_dictionary[agent_name[i]]

        res = agent_results[i]
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(res)
        x_vals = list(range(len(mean_results)))
        ax.plot(x_vals, mean_results, label=agent_name[i], color=color)
        # ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        # ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.3, color=color, linewidth=0)

    # ax.set_xlim([0, len(agent_results[0][0])])
    ax.set_ylim([y_min, y_max])
    # ax.yaxis.set_major_locator(x_major_locator)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    y_major_locator = MultipleLocator(3)
    ax.yaxis.set_major_locator(y_major_locator)
    x_major_locator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_major_locator)

    ax.tick_params(axis='both',
                   labelsize=16,  # y轴字体大小设置
                   width=1.5,
                   length=3
                   )
    legend_font = {
        'family': 'Times New Roman',  # 字体
        'size': 14

    }

    ax.legend(
        prop=legend_font,
        # frameon=False,
        loc='upper right',
        framealpha=0.8

    )

    plt.savefig(file_path_for_pic, dpi=600)


def visualize_delay_distribution(agent_results, agent_name: list, file_path_for_pic, y_min, y_max):
    """
    Visualize the results for one agent.
    :param y_label:
    :param tile:
    :param file_path_for_pic:
    :param agent_results: list of lists, each
    :param agent_name:
    :return:
    """
    # assert isinstance(agent_results, list), 'agent_results must be a list of list of list.'
    # assert isinstance(agent_results[0], list), 'agent_result must be a list of list of list.'
    # assert isinstance(agent_results[0][0], list), 'agent_result must be a list of list of list.'
    # x_major_locator = MultipleLocator(5)
    plt.rcParams['axes.unicode_minus'] = False
    agent_to_color_dictionary = {
        'continuous': '##0099DD',
        # 'discrete': '#FF69B4',
        # 'hybrid': '#800080',
        'sequence 1': '#0099DD',
        'sequence 2': '#FF69B4',
        'sequence 3': '#800080',
        'light': '#0099DD',
        'medium': '#FF69B4',
        'heavy': '#800080',
        '5 second': '#FF69B4',
        '10 second': '#0099DD',
        '15 second': '#F25A38',
        '20 second': '#800080',
        '25 second': '#032CA6',
        '30 second': '#D91E41',
        '11 second': '#FF69B4',
        '12 second': '#0099DD',
        '13 second': '#F25A38',
        '14 second': '#800080',
        '16 second': '#FF69B4',
        '17 second': '#0099DD',
        '18 second': '#F25A38',
        '19 second': '#800080',
        '21 second': '#FF69B4',
        '22 second': '#0099DD',
        '23 second': '#F25A38',
        '24 second': '#800080',
        'FRAP': '#800080',

        'discrete': '#FF69B4',
        'continuous opposite': '#0099DD',
        'hybrid': '#F25A38',
        'continuous single': '#800080',

        'maximum delay': '#800080',
        'average delay': '#0099DD',
        'queue length': '#FF69B4',
    }
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.set_ylabel('Proportion (%)', fontdict={'font': 'Times New Roman', 'fontsize': 14}, labelpad=6.0)
    ax.set_xlabel('Delay (s)', fontdict={'font': 'Times New Roman', 'fontsize': 14})
    # for spine in ['right', 'top']:
    # ax.spines[spine].set_visible(False)

    for i in range(2):
        color = agent_to_color_dictionary[agent_name[i]]

        res = agent_results[i]
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(res)
        x_vals = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 190]
        # ax.plot(x_vals, mean_results, label=agent_name[i], color=color)
        # ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        # ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax.fill_between(x_vals, y1=np.array([0]*len(mean_results)), y2=mean_results, alpha=0.5, color=color, linewidth=1.5)

    ax.set_xlim([0, 200])
    ax.set_ylim([y_min, y_max])
    # ax.yaxis.set_major_locator(x_major_locator)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    ax.tick_params(axis='both',
                   labelsize=14,  # y轴字体大小设置
                   width=1,
                   length=2.5
                   )
    legend_font = {
        'family': 'Times New Roman',  # 字体
        'size': 14

    }

    ax.legend(
        prop=legend_font,
        frameon=False,
        loc='lower right'
    )

    plt.tight_layout()
    plt.savefig(file_path_for_pic, dpi=600)


def visualize_overall_agent_results_twin(agent_results, agent_name: list, file_path_for_pic, tile, y_label_1, y_label_2,
                                         y_min_1, y_max_1, y_min_2, y_max_2):
    """
    Visualize the results for one agent.
    :param y_label:
    :param tile:
    :param file_path_for_pic:
    :param agent_results: list of lists, each
    :param agent_name:
    :return:
    """
    # assert isinstance(agent_results, list), 'agent_results must be a list of list of list.'
    # assert isinstance(agent_results[0], list), 'agent_result must be a list of list of list.'
    # assert isinstance(agent_results[0][0], list), 'agent_result must be a list of list of list.'
    # x_major_locator = MultipleLocator(5)
    plt.rcParams['axes.unicode_minus'] = False
    agent_to_color_dictionary = {
        'continuous': '##0099DD',
        # 'discrete': '#FF69B4',
        # 'hybrid': '#800080',
        'sequence 1': '#0099DD',
        'sequence 2': '#FF69B4',
        'sequence 3': '#800080',
        'light': '#0099DD',
        'medium': '#FF69B4',
        'heavy': '#800080',
        '5 second': '#FF69B4',
        '10 second': '#0099DD',
        '15 second': '#F25A38',
        '20 second': '#800080',
        '25 second': '#032CA6',
        '30 second': '#D91E41',
        '11 second': '#FF69B4',
        '12 second': '#0099DD',
        '13 second': '#F25A38',
        '14 second': '#800080',
        '16 second': '#FF69B4',
        '17 second': '#0099DD',
        '18 second': '#F25A38',
        '19 second': '#800080',
        '21 second': '#FF69B4',
        '22 second': '#0099DD',
        '23 second': '#F25A38',
        '24 second': '#800080',
        'FRAP': '#800080',

        'discrete': '#FF69B4',
        'continuous opposite': '#0099DD',
        'hybrid': '#F25A38',
        'continuous single': '#800080',

        'maximum delay': '#0099DD',
        'average delay': '#800080',
        'queue length': '#FF69B4',
    }
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.set_ylabel(ylabel=y_label_1, fontdict={'font': 'Times New Roman', 'fontsize': 14}, labelpad=6.0)
    ax.set_xlabel('Episode', fontdict={'font': 'Times New Roman', 'fontsize': 14})
    # for spine in ['right', 'top']:
    # ax.spines[spine].set_visible(False)
    ax2 = ax.twinx()
    ax.set_ylabel(ylabel=y_label_2, fontdict={'font': 'Times New Roman', 'fontsize': 14}, labelpad=6.0)

    for i in range(2):
        color = agent_to_color_dictionary[agent_name[i]]

        res = agent_results[i]
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(res)
        x_vals = list(range(len(mean_results)))
        ax.plot(x_vals, mean_results, label=agent_name[i], color=color, linestyle='--')
        # ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        # ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.3, color=color, linewidth=0)

    for i in range(1):
        color = agent_to_color_dictionary[agent_name[i + 2]]

        res = agent_results[i + 2]
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(res)
        x_vals = list(range(len(mean_results)))
        ax2.plot(x_vals, mean_results, label=agent_name[i + 2], color=color)
        # ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        # ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax2.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.3, color=color, linewidth=0)

    ax.set_xlim([0, len(agent_results[0][0])])
    ax.set_ylim([y_min_1, y_max_1])
    ax2.set_ylim([y_min_2, y_max_2])
    # ax.yaxis.set_major_locator(x_major_locator)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    x2_label = ax2.get_xticklabels()
    [x2_label_temp.set_fontname('Times New Roman') for x2_label_temp in x2_label]
    y2_label = ax2.get_yticklabels()
    [y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

    ax.tick_params(axis='both',
                   labelsize=14,  # y轴字体大小设置
                   width=1,
                   length=2.5
                   )
    legend_font = {
        'family': 'Times New Roman',  # 字体
        'size': 14

    }

    ax2.tick_params(axis='both',
                    labelsize=14,  # y轴字体大小设置
                    width=1,
                    length=2.5
                    )

    ax.legend(
        prop=legend_font,
        frameon=False,
        loc='lower right'
    )

    plt.tight_layout()
    plt.savefig(file_path_for_pic, dpi=600)


def visualize_overall_agent_results_broke(agent_results, agent_name: list, file_path_for_pic, tile, y_label, y_min,
                                          y_max):
    """
    Visualize the results for one agent.
    :param y_label:
    :param tile:
    :param file_path_for_pic:
    :param agent_results: list of lists, each
    :param agent_name:
    :return:
    """
    # assert isinstance(agent_results, list), 'agent_results must be a list of list of list.'
    # assert isinstance(agent_results[0], list), 'agent_result must be a list of list of list.'
    # assert isinstance(agent_results[0][0], list), 'agent_result must be a list of list of list.'

    agent_to_color_dictionary = {
        'continuous': '#0000FF',
        'discrete': '#FF69B4',
        'hybrid': '#800080',
        'pattern_1': '#0000FF',
        'pattern_2': '#FF69B4',
        'pattern_3': '#800080',

    }

    fig = plt.subplots()
    bax = brokenaxes(xlims=((0, 1), (2, len(agent_results[0][0]))), ylims=((y_min, 70.), (130., y_max)))
    bax.set_facecolor('xkcd:white')
    bax.set_ylabel(y_label, fontdict={'font': 'Times New Roman', 'fontsize': 10}, labelpad=6.0)
    bax.set_xlabel('Episode', fontdict={'font': 'Times New Roman', 'fontsize': 10})
    # for spine in ['right', 'top']:
    #     bax.spines[spine].set_visible(False)

    for i in range(2):
        color = agent_to_color_dictionary[agent_name[i]]

        res = agent_results[i]
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(res)
        x_vals = list(range(len(mean_results)))
        bax.plot(x_vals, mean_results, label=agent_name[i], color=color)
        # ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        # ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        bax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.3, color=color, linewidth=0)

    bax.legend(loc='upper right', frameon=False)

    """x1_label = bax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = bax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]"""

    bax.tick_params(axis='both',
                    labelsize='medium',  # y轴字体大小设置
                    width=1,
                    length=2.5
                    )
    legend_font = {
        'family': 'Times New Roman',  # 字体
    }

    bax.legend(
        prop=legend_font,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(file_path_for_pic, dpi=600)


def get_mean_and_standard_deviation_difference(results):
    """
    From a list of lists of specific agent results it extracts the mean result and the mean result plus or minus
    some multiple of standard deviation.
    :param results:
    :return:
    """

    def get_results_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return results_at_a_time_step

    def get_std_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return np.std(results_at_a_time_step)

    mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
    mean_minus_x_std = [mean_val - get_std_at_a_time_step(results, timestep)
                        for timestep, mean_val in enumerate(mean_results)]
    mean_plus_x_std = [mean_val + get_std_at_a_time_step(results, timestep)
                       for timestep, mean_val in enumerate(mean_results)]
    return mean_minus_x_std, mean_results, mean_plus_x_std


def get_y_limits(results):
    """
    Extracts the minimum and maximum seen y_vals from a set of results.
    :param results:
    :return:
    """
    res_flattened = np.array(results).flatten()
    max_res = np.max(res_flattened)
    min_res = np.min(res_flattened)
    y_limits = [min_res - 0.05 * (max_res - min_res), max_res + 0.05 * (max_res - min_res)]

    return y_limits


def visualize_results_per_run(agent_results, agent_name, file_path_for_pic, y_label):
    """
    :param y_label:
    :param file_path_for_pic:
    :param agent_name:
    :param agent_results:
    :return:
    """
    assert isinstance(agent_results, list), 'agent_results must be a list'
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.set_ylabel(y_label)
    ax.set_xlabel('Episode')

    agent_to_color_dictionary = {
        'continuous': '#0000FF',
        'discrete': '#800080',
        'hybrid': '#FF69B4',
        'hyar': '#F25A38',
        'MaxPressure': '#737373',
        'FRAP': '#800080',
    }
    color = agent_to_color_dictionary[agent_name]

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    x_vals = list(range(len(agent_results)))
    ax.set_xlim([0, x_vals[-1]])
    ax.set_ylim([min(agent_results), max(agent_results)])
    ax.plot(x_vals, agent_results, label=agent_name, color=color)
    ax.legend(loc='upper right', shadow='Ture', facecolor='inherit')
    plt.tight_layout()
    plt.savefig(file_path_for_pic)
    plt.close('all')


def plot_flow_heatmap(data):
    """

    """

    # movement labels
    movement_labels = ["North straight", "east straight", "south straight", "west straight",
                       "north left", "east left", "south left", "west left"]
    # internal labels
    internal_labels = ["0", "", "15", "", "30", "", "45", "", "60"]

    # plot
    fig, ax = plt.subplots()

    # spines
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Time of day (min)', fontdict={'font': 'Times New Roman', 'fontsize': 10})

    im = ax.imshow(data, alpha=1, cmap='Oranges', extent=[-0.5, 7.5, 8, 0])
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Flow (veh/h)', labelpad=16.0, rotation=90, va="bottom",
                       fontdict={'font': 'Times New Roman', 'fontsize': 10})
    cbar.ax.tick_params(
        labelsize='medium',
        width=1,
        length=2.5,
    )
    cbar.outline.set_visible(False)

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(movement_labels)), movement_labels)
    ax.set_yticks(np.arange(len(internal_labels)), internal_labels)

    ax.invert_yaxis()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Set the font name of the tick labels
    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    ax.tick_params(axis='both',
                   labelsize='medium',  # y轴字体大小设置
                   width=1,
                   length=2.5
                   )

    fig.tight_layout()
    plt.show()


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def extract_rolling_score(record_mark, action_space_pattern, worker_idx):
    indicator = 0
    for idx in worker_idx:
        raw_delay = np.load('data/rolling_data/{}/{}/delay_{}.npz'.format(record_mark, action_space_pattern, idx))
        raw_queue = np.load('data/rolling_data/{}/{}/queue_{}.npz'.format(record_mark, action_space_pattern, idx))
        if indicator == 0:
            delay = np.array([raw_delay['delay']])
            queue = np.array([raw_queue['queue']])
            indicator = 1
        else:
            delay = np.append(delay, [raw_delay['delay']], axis=0)
            queue = np.append(queue, [raw_queue['queue']], axis=0)
    return delay, queue, delay.shape[1]


def extract_over_all_rs(record_mark, worker_idx: list):
    delay_con, queue_con, size_con = extract_rolling_score(record_mark, 'continuous', worker_idx)
    delay_dis, queue_dis, size_dis = extract_rolling_score(record_mark, 'discrete', worker_idx)
    delay_hybrid, queue_hybrid, size_hybrid = extract_rolling_score(record_mark, 'hybrid', worker_idx)
    min_size = min(size_con, size_dis, size_hybrid)
    delay = np.array([
        delay_con[:, :min_size],
        delay_dis[:, :min_size],
        delay_hybrid[:, :min_size]
    ])

    queue = np.array([
        queue_con[:, :min_size],
        queue_dis[:, :min_size],
        queue_hybrid[:, :min_size]
    ])

    return delay, queue


class Monitor:
    """
    A monitor in training and evaluating process.

    """

    def __init__(self, rolling_score_window):
        self.monitor_0 = {'queue': [], 'delay': []}
        self.monitor_1 = {'queue': [], 'delay': []}
        self.monitor_2 = {'queue': [], 'delay': []}  # rolling score
        self.rolling_score_window = rolling_score_window

    def push_into_monitor(self, queue, delay):
        self.monitor_0['queue'].append(queue)
        self.monitor_0['delay'].append(delay)

    def output_from_monitor(self):
        self.monitor_1['queue'].append(np.mean(self.monitor_0['queue']))
        self.monitor_1['delay'].append(np.mean(self.monitor_0['delay']))
        self.monitor_2['queue'].append(np.mean(self.monitor_1['queue'][-1 * self.rolling_score_window:]))
        self.monitor_2['delay'].append(np.mean(self.monitor_1['delay'][-1 * self.rolling_score_window:]))

        self.monitor_0 = {'queue': [], 'delay': []}

        return self.monitor_1, self.monitor_2

    def output(self):
        return self.monitor_0


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)
        

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # print("x",x)
    # print("y",y)

    x_norm = (x ** 2).sum(1).view(-1, 1)   #sum(1)将一个矩阵的每一行向量相加
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print("x_norm",x_norm)
    # print("y_norm",y_norm)
    y_t = torch.transpose(y, 0, 1)  #交换一个tensor的两个维度
    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    #torch.mm 矩阵a和b矩阵相乘
    # dist[dist != dist] = 0 # replace nan values with 0
    # print("dist",dist)
    return dist


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def custom_weight_init(self):
        # Initialize the weight values
        for m in self.modules():
            weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return
