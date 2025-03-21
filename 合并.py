import h5py
import numpy as np
import matplotlib.pyplot as plt

# 文件路径定义
earthquake_hdf5_file = "D:\\地震数据\\new_data\\filtered_chunk4.hdf5"
output_hdf5_file = "C:\\Users\\admin\\OneDrive\\Desktop\\combined_data.hdf5"

def plot_time_series_comparison(hdf5_earthquake_path, hdf5_combined_path):
    # 读取HDF5文件中的合并数据
    with h5py.File(hdf5_combined_path, 'r') as hdf5_combined, \
            h5py.File(hdf5_earthquake_path, 'r') as hdf5_earthquake:

        combined_datasets = list(hdf5_combined['data'].keys())
        earthquake_datasets = list(hdf5_earthquake['data'].keys())

        if not combined_datasets or not earthquake_datasets:
            print("No datasets found in the HDF5 file(s).")
            return

        # 统计合并后的波形数据集数量
        combined_waveform_count = len(combined_datasets)

        for dataset_name in combined_datasets:
            if dataset_name.startswith('combined_'):
                index = int(dataset_name.split('_')[1])
                original_dataset_name = f'data/{earthquake_datasets[index]}'

                if original_dataset_name in hdf5_earthquake:
                    # 获取原始地震数据
                    original_data = np.array(hdf5_earthquake[original_dataset_name])

                    # 获取合并后的数据
                    combined_data = np.array(hdf5_combined[f'data/{dataset_name}'])

                    # 创建一个新的图形
                    plt.figure(figsize=(14, 6))

                    # 如果数据是多分量（例如3个分量），为每个分量创建子图
                    if len(original_data.shape) > 1 and len(combined_data.shape) > 1:
                        num_components = original_data.shape[1]
                        for comp in range(num_components):
                            plt.subplot(num_components, 1, comp + 1)
                            plt.plot(original_data[:, comp], label='Original Earthquake Signal', color='blue')
                            plt.plot(combined_data[:, comp], label='Combined Signal', color='black', linestyle='-')
                            plt.title(f'Comparison {dataset_name} - Component {comp + 1}')
                            plt.xlabel('Sample Index')
                            plt.ylabel('Amplitude')
                            plt.grid(True)
                            plt.legend()
                    else:
                        # 创建单一分量的时间序列图
                        plt.plot(original_data, label='Original Earthquake Signal', color='blue')
                        plt.plot(combined_data, label='Combined Signal', color='black', linestyle='-')
                        plt.title(f'Comparison {dataset_name}')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Amplitude')
                        plt.grid(True)
                        plt.legend()

                    # 调整布局以防止重叠
                    plt.tight_layout()

                    # 显示图形并等待用户关闭窗口
                    plt.show()
                else:
                    print(f"Warning: Original dataset for {dataset_name} not found.")

        # 打印合并后的波形数据集数量
        print(f"Total number of combined waveforms: {combined_waveform_count}")

# 调用函数绘制时间序列对比图并打印合并后波形数据集的数量
plot_time_series_comparison(earthquake_hdf5_file, output_hdf5_file)