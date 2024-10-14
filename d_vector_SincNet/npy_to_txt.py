# import numpy as np
# import json
# test=np.load('/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/d_vect_timit.npy',encoding = "latin1", allow_pickle=True) #加载文件
# # doc = open('d_vect_timit.txt', "w") #打开一个存储文件，并依次写入
# print(str(test)[:100])
# json_output = json.loads(str(test))

# b = json.dumps(json_output)
# f2 = open('new_json.json', 'w')
# f2.write(b)
# f2.close()

import numpy as np
from collections import Counter
# 加载 .npy 文件
file_path = '/home/cuizhouying/CS4347/CS4347/d_vector_SincNet/data_lists/TIMIT_labels.npy'
data = np.load(file_path, allow_pickle=True)

# 访问零维数组中包含的对象
loaded_object = data.item()

unique_values_count = len(set(loaded_object.values()))
print(f"Number of unique values: {unique_values_count}")
# # 查看数据类型
# print(f"Loaded object type: {type(loaded_object)}")

# # 如果是字典，可以查看它的内容
# if isinstance(loaded_object, dict):
#     print(f"Keys: {list(loaded_object.keys())}")
#     for key, value in loaded_object.items():
#         print(f"Key: {key}, Value shape: {value.shape}, Value type: {type(value)}")

