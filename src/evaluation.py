import time
import scipy.io as sio
import sys

sys.path.append("../")
# from baseline import chinese_whispers as cw
from baseline import aroc as ac
from metric import f1_score as assess

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	
	# 利用第三方测试工具
	data = sio.loadmat("../dataset/LightenedCNN_C_lfw.mat")
	features = data['features']
	# print(features.shape)
	# print(type(features))
	labels = data['labels_original'][0]
	label_lookup = {}
	for idx, label in enumerate(labels):
		label_lookup[idx] = int(label[0][:])
	# print('Features shape: ', features.shape)
	
	start_time = time.time()
	# face_encodings = {}
	# for path, embeddings in zip(data['image_path'][0], features):
	# 	# print(path[0], embeddings)
	# 	face_encodings[path[0]] = embeddings
	# clusters = cw.cluster_face_encodings(face_encodings, threshold=2.0, iterations=10)
	clusters = ac.aroc(features, 200, 1.5, num_proc=4)
	print('Time taken for clustering: {:.3f} seconds'.format(
		time.time() - start_time))
	
	precision, recall, f1_score = assess.f1_score(clusters, label_lookup)
	print(precision, recall, f1_score)
