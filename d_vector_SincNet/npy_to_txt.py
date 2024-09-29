import numpy as np
import json
test=np.load('d_vector_SincNet\d_vect_timit.npy',encoding = "latin1", allow_pickle=True) #加载文件
# doc = open('d_vect_timit.txt', "w") #打开一个存储文件，并依次写入
print(str(test)[:100])
json_output = json.loads(str(test))

b = json.dumps(json_output)
f2 = open('new_json.json', 'w')
f2.write(b)
f2.close()