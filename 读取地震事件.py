import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_name ="D:\dizhenshujuji\stead_data\chunk2.hdf5"
csv_file = "D:\dizhenshujuji\stead_data\chunk2.csv"

# 读取CSV文件到DataFrame
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')

# # 筛选DataFrame
# df = df[(df.trace_category == 'earthquake_local') &
#         # (df.source_distance_km <= 20) &
#         # (df.source_magnitude > 3) &
#         (df.p_weight == 1)]
#
# print(f'total events selected: {len(df)}')

# 为筛选后的数据生成轨迹名称列表
ev_list = df['trace_name'].to_list()

# 从HDF5文件中检索选定的波形数据
dtfl = h5py.File(file_name, 'r')
for c, evi in enumerate(ev_list):
    dataset = dtfl.get('data/' + str(evi))
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

    inp = input("Press a key to plot the next waveform!")
    if inp == "r":
        continue
