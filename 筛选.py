import os
import pandas as pd
import h5py
import numpy as np

# 文件路径定义
hdf5_file = "D:\\dizhenshujuji\\stead_data\\chunk4.hdf5 "
csv_file = "D:\\dizhenshujuji\\stead_data\\chunk4.csv"
filtered_csv_file = "D:\\地震数据\\new_data\\filtered_chunk4.csv"
output_hdf5_file = "D:\地震数据\\new_data\\zfiltered_chunk4.hdf5"

try:
    # 确保目标目录存在
    output_dir = os.path.dirname(filtered_csv_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取CSV文件到DataFrame
    df = pd.read_csv(csv_file, low_memory=False)  # 添加 low_memory=False 以避免 dtype 警告
    print(f'Total events in CSV file: {len(df)}')

    # 筛选DataFrame
    df_filtered = df[(df.trace_category == 'earthquake_local') &
                     (df.s_weight <= 0.5) &
                     (df.p_weight <= 0.5) &
                     df['snr_db'].apply(lambda x: all(float(i) > 40 for i in x.strip('[]').split()))]

    # 限制为前10条记录
    df_filtered = df_filtered.head(10)

    # 将筛选后的DataFrame保存到新的CSV文件
    df_filtered.to_csv(filtered_csv_file, index=False)
    print(f'Successfully saved filtered data to {filtered_csv_file}')

    # 获取筛选后的trace_name列表
    selected_trace_names = df_filtered['trace_name'].tolist()

    # 打开原始HDF5文件和创建新的HDF5文件
    with h5py.File(hdf5_file, 'r') as hdf5_in, h5py.File(output_hdf5_file, 'w') as hdf5_out:
        for trace_name in selected_trace_names:
            dataset_path = f'data/{trace_name}'
            if dataset_path in hdf5_in:
                dataset = hdf5_in[dataset_path]
                data = np.array(dataset)  # 获取数据

                # 如果需要对数据进行任何处理（如标准化），可以在下面添加代码
                # 例如：data = process_data(data)

                # 将数据保存到新的HDF5文件
                hdf5_out.create_dataset(dataset_path, data=data)
                # 复制原有属性
                for key, value in dataset.attrs.items():
                    hdf5_out[dataset_path].attrs[key] = value
            else:
                print(f"Warning: Dataset for trace_name {trace_name} not found.")

    print(f'Successfully saved filtered HDF5 data to {output_hdf5_file}')

except Exception as e:
    print(f'An error occurred: {e}')