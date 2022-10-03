import loaddata
import numpy as np
data_dir='./att_align'

data = loaddata.read_data_sets(data_dir)
train=data.train
print(train)

for i in range(100):
	www,label = train.next_batch(64)
	print(np.array(www).shape)