import numpy as np

data = np.load('../dataset/val/SCH_001/CLS_0024/P005497/video/qc_stats/A01/sequence.npz')

print("文件中的键名:", data.files)
for key in data.files:
    print(f"\n--- 键名: {key} ---")
    array = data[key]
    print(f"形状 (Shape): {array.shape}")
    if array.size > 100:
        print("内容 (前5个元素):", array.flat[:5])
        print("...")
    else:
        print("内容:", array)

data.close()
