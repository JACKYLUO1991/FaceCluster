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
from sklearn import manifold
import sys
import os
import os.path as osp

from imutils import paths
from facenet_pytorch import MTCNN, InceptionResnetV1


if __name__ == '__main__':
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	mtcnn = MTCNN(device=device)
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	
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
			# features.append(embeddings[0])
			# print(x_aligned.cpu().numpy().transpose((1, 2, 0)))
			# plt.imshow(x_aligned.cpu().numpy().transpose((1, 2, 0)))
			# plt.show()
			# sys.exit(-1)
			
			# 给定图像元信息
			info = [{'imagePath': str(img_path), "encodings": embeddings}]
			features.extend(info)
		else:
			ignore_imgs.append(str(img_path))
	
	# # t-SNE是一种集降维与可视化与一体的技术
	# tsne = manifold.TSNE(n_components=2, init='pca', random_state=2019)
	# transformed = tsne.fit_transform(features)
	#
	# plt.scatter(transformed[:, 0], transformed[:, 1], c='r')
	# plt.show()
	#
	features = np.array(features)
	print("Ignore image: ", ignore_imgs)
	#
	# # print(features)
	# # aligned = torch.stack(aligned).to(device)
	# # features = resnet(aligned).cpu()
	#
	# # dists = [[(e1 - e2).norm().item() for e2 in features] for e1 in features]
	# # df = pd.DataFrame(dists, columns=np.arange(0, len(features)), index=np.arange(0, len(features)))
	# # corrs = df.corr()
	# #
	# # import plotly.figure_factory as ff
	# # import plotly.offline as py
	# # figure = ff.create_annotated_heatmap(z=corrs.values, x=list(corrs.columns), y=list(corrs.index), annotation_text=corrs.round(2).values, showscale=True)
	# # py.plot(figure)
	#
	
	import sys
	sys.path.append("../")
	from baseline import aroc as ac

	# Approximate Rank-order clustering
	# TODO: 自己的调参经验n_neighbours与每个簇的样本数有关, threshold还在想如何???
	encodings = [d["encodings"][0] for d in features]
	imagePaths = [i['imagePath'] for i in features]
	clusters = ac.aroc(np.array(encodings), 5, 0.9, num_proc=4)
	
	# 字典类型的图像聚类结果
	clusters_index = {}
	for i, cluster in enumerate(clusters):
		clusters_index[i] = [int(x) for x in list(cluster)]
		
	for key, values in clusters_index.items():
		for value in values:
			if not osp.exists(osp.join("../resource", str(key))):
				os.mkdir(osp.join("../resource", str(key)))
			im = Image.open(imagePaths[value])
			im.save(osp.join("../resource", str(key), "{}.png").format(value))