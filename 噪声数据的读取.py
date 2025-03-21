import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_name = "D:\\dizhenshujuji\\stead_data\\chunk1.hdf5 "  # 注意这里使用了双反斜杠来转义路径中的反斜杠
csv_file = "D:\\dizhenshujuji\\stead_data\\chunk1.csv"

# 读取CSV文件到DataFrame
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')

# 筛选DataFrame（如果需要的话）
# df = df[(df.trace_category == 'earthquake_local') &
#         (df.p_weight == 1)]
# print(f'total events selected: {len(df)}')

# 为筛选后的数据生成轨迹名称列表
ev_list = df['trace_name'].to_list()

# 从HDF5文件中检索选定的波形数据
dtfl = h5py.File(file_name, 'r')
counter = 0  # 添加计数器

for c, evi in enumerate(ev_list):
    if counter >= 10:  # 当计数器达到10时，停止循环
        break
    dataset = dtfl.get('data/' + str(evi))
    if dataset is None:
        continue  # 如果dataset不存在，则跳过这个事件
    # 波形数据，3个通道：第一个行是E通道，第二行是N通道，第三行是Z通道
    data = np.array(dataset)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(312)
    plt.plot(data[:, 1], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(313)
    plt.plot(data[:, 2], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight': 'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc='upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])
    plt.show()

    for at in dataset.attrs:
        print(at, dataset.attrs[at])

    inp = input("Press a key to plot the next waveform! Type 'r' to skip to the next one.")
    if inp == "r":
        continue
    counter += 1  # 每次循环结束时增加计数器