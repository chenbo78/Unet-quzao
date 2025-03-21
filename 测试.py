import h5py
import numpy as np
import logging
from tqdm import tqdm  # 进度条库

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义文件路径
noise_hdf5_file = "D:\\dizhenshujuji\\stead_data\\chunk1.hdf5"
earthquake_hdf5_file = "D:\\地震数据\\new_data\\filtered_chunk4.hdf5"
output_combined_file = "C:\\Users\\admin\\OneDrive\\Desktop\\combined_data.hdf5"

# 提取前N个噪声波形数据
def extract_noise_waveforms(input_file, N):
    waveforms = []
    try:
        with h5py.File(input_file, 'r') as hdf:
            datasets = sorted(list(hdf['data'].keys()))[:N]
            for name in tqdm(datasets, desc=f"Loading {N} noise waveforms from {input_file}"):
                waveforms.append(hdf['data'][name][:])
        logging.info(f"Loaded {len(waveforms)} noise waveforms.")
    except Exception as e:
        logging.error(f"Error loading noise waveforms from {input_file}: {e}")
    return np.array(waveforms)


# 将噪声加到地震波形上并保存到新的HDF5文件
def combine_and_save_waveforms(noise_waveforms, earthquake_file, output_file):
    try:
        with h5py.File(earthquake_file, 'r') as eq_hdf, h5py.File(output_file, 'w') as combined_hdf:
            eq_datasets = list(eq_hdf['data'].keys())
            if len(eq_datasets) < len(noise_waveforms):
                logging.warning("Fewer earthquake waveforms than noise waveforms. Using all available earthquake waveforms.")

            for i, (eq_name, noise) in enumerate(zip(tqdm(eq_datasets[:len(noise_waveforms)], desc="Combining waveforms"), noise_waveforms)):
                eq_waveform = eq_hdf['data'][eq_name][:]
                combined_waveform = eq_waveform + noise

                # 创建新数据集并复制原有属性
                new_dataset_name = f'data/waveform_{i}'
                combined_hdf.create_dataset(new_dataset_name, data=combined_waveform)
                original_dataset = eq_hdf['data'][eq_name]
                for key, value in original_dataset.attrs.items():
                    combined_hdf[new_dataset_name].attrs[key] = value

        logging.info(f"All combined waveforms saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error combining and saving waveforms: {e}")


# 主程序
def main():
    N = 10  # 提取前10个噪声波形

    # 抽取噪声波形数据
    noise_waveforms = extract_noise_waveforms(noise_hdf5_file, N)

    # 将噪声加到地震波形上并保存
    combine_and_save_waveforms(noise_waveforms, earthquake_hdf5_file, output_combined_file)


if __name__ == "__main__":
    main()