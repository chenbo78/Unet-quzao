from tensorflow.keras.regularizers import l1_l2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
# 文件路径定义
earthquake_hdf5_file ="D:\\地震数据\\new_data\\filtered_chunk4.hdf5"
output_hdf5_file ="C:\\Users\\admin\\OneDrive\\Desktop\\combined_data.hdf5"



def load_and_preprocess_data(hdf5_combined_path, hdf5_earthquake_path):
    combined_waveforms = []
    original_waveforms = []

    with h5py.File(hdf5_combined_path, 'r') as hdf5_combined, \
            h5py.File(hdf5_earthquake_path, 'r') as hdf5_earthquake:
        combined_datasets = list(hdf5_combined['data'].keys())
        earthquake_datasets = list(hdf5_earthquake['data'].keys())

        print(f"Combined datasets: {len(combined_datasets)}")
        print(f"Earthquake datasets: {len(earthquake_datasets)}")

        # 假设两个文件中的键名格式相同且顺序一致
        for dataset_name in combined_datasets:
            if dataset_name.startswith('combined_'):
                index = int(dataset_name.split('_')[1])
                if index < len(earthquake_datasets):
                    original_dataset_name = earthquake_datasets[index]
                    if original_dataset_name.startswith('109C.TA_'):
                        original_waveforms.append(np.array(hdf5_earthquake['data/' + original_dataset_name]))
                        combined_waveforms.append(np.array(hdf5_combined['data/' + dataset_name]))
                    else:
                        print(f"Original dataset name does not start with '109C.TA_': {original_dataset_name}")
                else:
                    print(f"Index {index} out of range for earthquake_datasets.")

    print(f"Loaded combined waveforms: {len(combined_waveforms)}")
    print(f"Loaded original waveforms: {len(original_waveforms)}")

    return np.array(combined_waveforms), np.array(original_waveforms)

# 测试数据加载
combined_waveforms, original_waveforms = load_and_preprocess_data(output_hdf5_file, earthquake_hdf5_file)
print(f"Combined waveforms shape: {combined_waveforms.shape}")
print(f"Original waveforms shape: {original_waveforms.shape}")

def build_unet_model(input_shape, num_features):
    inputs = Input(shape=input_shape)

    # Encoder (downsampling)
    conv1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    bn1 = BatchNormalization()(conv1)
    dropout1 = Dropout(0.6)(bn1)
    pool1 = MaxPooling1D(pool_size=2)(dropout1)

    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pool1)
    bn2 = BatchNormalization()(conv2)
    dropout2 = Dropout(0.6)(bn2)
    pool2 = MaxPooling1D(pool_size=2)(dropout2)

    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pool2)
    bn3 = BatchNormalization()(conv3)
    dropout3 = Dropout(0.6)(bn3)
    pool3 = MaxPooling1D(pool_size=2)(dropout3)

    # Bottleneck
    convb = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pool3)
    bnb = BatchNormalization()(convb)
    dropoutb = Dropout(0.6)(bnb)

    # Decoder (upsampling)
    up1 = UpSampling1D(size=2)(dropoutb)
    concat3 = concatenate([up1, dropout3], axis=-1)  # Skip connection
    conv4 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(concat3)
    bn4 = BatchNormalization()(conv4)
    dropout4 = Dropout(0.6)(bn4)

    up2 = UpSampling1D(size=2)(dropout4)
    concat2 = concatenate([up2, dropout2], axis=-1)  # Skip connection
    conv5 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(concat2)
    bn5 = BatchNormalization()(conv5)
    dropout5 = Dropout(0.6)(bn5)

    up3 = UpSampling1D(size=2)(dropout5)
    concat1 = concatenate([up3, dropout1], axis=-1)  # Skip connection
    conv6 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(concat1)
    bn6 = BatchNormalization()(conv6)
    dropout6 = Dropout(0.6)(bn6)

    # Output layer to match the shape of the input signal
    outputs = Conv1D(filters=num_features, kernel_size=3, activation='linear', padding='same')(dropout6)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mean_absolute_error')  # 使用MAE作为损失函数

    return model



def lr_scheduler(epoch, lr):
    if epoch > 10:
        return lr * 0.1
    return lr

lr_scheduler = LearningRateScheduler(lr_scheduler)


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # 减少patience值
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)  # 减少patience值

    history = model.fit(X_train, y_train, epochs=32, batch_size=64, shuffle=True,
                        validation_split=0.1, callbacks=[early_stopping, reduce_lr,lr_scheduler])

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    print(f"Mean Squared Error on Test Set: {mse}")
    print(f"Mean Absolute Error on Test Set: {mae}")

    return history, predictions, mse, mae

def main():
    # 加载并预处理数据
    combined_waveforms, original_waveforms = load_and_preprocess_data(output_hdf5_file, earthquake_hdf5_file)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(combined_waveforms, original_waveforms, test_size=0.2,
                                                        random_state=42)

    # 确保输入形状正确 (samples, time steps, features)
    if len(X_train.shape) == 2:
        num_features = 1  # 如果是单通道
    else:
        num_features = X_train.shape[2]  # 如果是多通道

    # 数据标准化
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)  # 计算整个训练集的均值
    X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-10  # 防止除零

    # 应用标准化到所有数据
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - X_mean) / X_std
    y_test = (y_test - X_mean) / X_std

    # 确保数据形状正确
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], num_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], num_features))

    # 构建UNet模型
    input_shape = (X_train.shape[1], num_features)
    unet_model = build_unet_model(input_shape, num_features)

    # 训练并评估模型
    history, predictions, mse, mae = train_and_evaluate_model(unet_model, X_train, y_train, X_test, y_test)
    # 可视化结果
    for i in range(min(5, len(X_test))):  # 只显示前5个样本的结果
        plt.figure(figsize=(14, 6))
        for comp in range(num_features):
            # 注意这里要确保只使用对应通道的均值和标准差进行反标准化
            plt.subplot(num_features, 1, comp + 1)
            plt.plot(y_test[i, :, comp].flatten() * X_std[0, 0, comp] + X_mean[0, 0, comp],
                     label=f'Original Earthquake Signal - Comp {comp + 1}', color='blue')
            plt.plot(predictions[i, :, comp].flatten() * X_std[0, 0, comp] + X_mean[0, 0, comp],
                     label=f'Denoised Signal - Comp {comp + 1}', color='red')
            plt.title(f'Comparison - Test Sample {i + 1}, Component {comp + 1}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()