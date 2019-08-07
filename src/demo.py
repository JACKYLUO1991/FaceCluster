# 利用最新的人脸识别算法 + 对齐操作
# https://github.com/timesler/facenet-pytorch/blob/master/models/utils/example.py
from PIL import Image
import time
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from imutils import paths
from facenet_pytorch import MTCNN, InceptionResnetV1


if __name__ == '__main__':
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	mtcnn = MTCNN(device=device)
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	
	# aligned = []
	features = []
	ignore_imgs = []
	for img_path in tqdm(list(paths.list_images("../dataset/"))):
		start = time.time()
		img = Image.open(img_path)
		# MTCNN需要输入PIL类型
		x_aligned = mtcnn(img)
		if x_aligned is not None:
			# 备注: 后期人脸检测不到的可以直接在文件中删除对应的图片
			embeddings = resnet(x_aligned.unsqueeze(0)).to(device).detach().numpy()
			# aligned.append(x_aligned)
			features.append(embeddings[0])
		else:
			ignore_imgs.append(str(img_path))
	features = np.array(features)
	print(ignore_imgs)
	
	# print(features)
	# aligned = torch.stack(aligned).to(device)
	# features = resnet(aligned).cpu()
	
	# dists = [[(e1 - e2).norm().item() for e2 in features] for e1 in features]
	# df = pd.DataFrame(dists, columns=np.arange(0, len(features)), index=np.arange(0, len(features)))
	# corrs = df.corr()
	#
	# import plotly.figure_factory as ff
	# import plotly.offline as py
	# figure = ff.create_annotated_heatmap(z=corrs.values, x=list(corrs.columns), y=list(corrs.index), annotation_text=corrs.round(2).values, showscale=True)
	# py.plot(figure)

	import sys
	sys.path.append("../")
	from baseline import aroc as ac
	
	# Approximate Rank-order clustering
	# TODO: 自己的调参经验n_neighbours与每个簇的样本数有关, threshold还在想如何???
	clusters = ac.aroc(features, 3, 0.9, num_proc=4)
	print(clusters)
	clusters_index = {}
	for i, cluster in enumerate(clusters):
		clusters_index[i] = [int(x) for x in list(cluster)]
	# print(clusters_index)
	
	# 结果保留在json文件中
	with open("clusters.json", "w", encoding='utf-8') as f:
		f.write(json.dumps(clusters_index, indent=4))
