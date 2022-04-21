import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import datetime
import argparse
import json

def main(conf, clustering_algorithm):

    model_folder = config['trainer']['save_dir'] + config['experim_name']
    ### VOC Dataset
    if config['dataset'] == 'voc':
        label_folder = 'datasets/voc/VOCdevkit/VOC2012/SegmentationClassAug'
        if conf['n_labeled_examples'] == 662:
            split_list = [132, 2, 1, 1, 1, 2, 3, 4, 7, 2, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 331:
            split_list = [121, 2, 1, 1, 1, 1, 3, 3, 6, 3, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 165:
            split_list = [136, 2, 2, 1, 1, 1, 2, 4, 8, 3, 1, 2, 7, 2, 2, 18, 1, 1, 1, 3, 3]
    ### Cityscapes Dataset
    elif config['dataset'] == 'cityscapes':
        label_folder = 'datasets/cityscapes/segmentation/train'
        if conf['n_labeled_examples'] == 372:
            split_list = [42, 7, 26, 1, 2, 2, 1, 1, 19, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 186:
            split_list = [45, 7, 28, 1, 2, 2, 1, 1, 20, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 93:
            split_list = [38, 6, 22, 1, 2, 2, 1, 1, 17, 2, 5, 1, 1, 7, 1, 1, 1, 1, 1]

    save_folder = os.path.join(model_folder, 'label_subcls_' + clustering_algorithm)
    os.makedirs(save_folder, exist_ok=True)
    feature_folder = os.path.join(model_folder, 'features')
    subclasses =  np.cumsum(np.asarray(split_list))
    subclasses = np.insert(subclasses, 0, 0)
    oldtime=datetime.datetime.now()
    files = os.listdir(feature_folder)
    list.sort(files)
    feat_shape_list = []
    label_shape_list = []
    for i, file in enumerate(files):
        feat = np.load(os.path.join(feature_folder, file))
        feat_shape_list.append(feat.shape[-2:])
        H, W = feat.shape[-2], feat.shape[-1]
        if config['dataset'] == 'cityscapes':
            label = np.asarray(Image.open(os.path.join(label_folder, file.replace('_leftImg8bit.npy', '_gtFine_labelTrainIds.png'))))
        else:
            label = np.asarray(Image.open(os.path.join(label_folder, file.replace('.npy', '.png'))))
        label_shape_list.append(label.shape[-2:])
        target = cv2.resize(label, (W,H), interpolation=cv2.INTER_NEAREST)
        feat = feat.reshape(feat.shape[0],-1).transpose((1,0))
        target = np.expand_dims(target.reshape(-1), axis=1)
        if i==0:
            feats = feat
            targets = target
            file_id = i * np.ones(target.shape)
        else:
            feats = np.vstack((feats, feat))
            targets = np.vstack((targets, target))
            file_id = np.vstack((file_id, i * np.ones(target.shape)))

    newtime=datetime.datetime.now()
    print('data_loading：%s'%(newtime-oldtime))
    print(Counter(targets.reshape(-1).tolist()))

    if clustering_algorithm == 'normal_kmeans':
        targets_subcls = targets.copy()
        for cls in np.unique(targets):
            print('Parent class:', cls)
            oldtime = datetime.datetime.now()
            if cls < 255:
                num_clusters = split_list[cls]
                subcls = subclasses[cls]
                if num_clusters == 1:
                    targets_subcls[targets==cls] = subcls
                else:
                    subindex = np.where(targets==cls)[0]
                    subfeats = feats[subindex,:]
                    k_center = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(subfeats)
                    newtime = datetime.datetime.now()
                    print('KMeans：%s' % (newtime - oldtime))
                    lbls = k_center.labels_
                    for j in range(num_clusters):
                        targets_subcls[subindex[lbls==j]] = subcls + j
    elif clustering_algorithm == 'balanced_kmeans':
        targets_subcls = targets.copy()
        for cls in np.unique(targets):
            print('Parent class:', cls)
            if cls < 255:
                num_clusters = split_list[cls]
                subcls = subclasses[cls]
                if num_clusters == 1:
                    targets_subcls[targets == cls] = subcls
                else:
                    subindex = np.where(targets == cls)[0]
                    subfeats = feats[subindex, :]

                    data_int16_x1000 = np.int16(subfeats * 1000)
                    np.savetxt(save_folder + '/subfeats_cls' + str(cls) + '_n' + str(num_clusters) + '.csv', data_int16_x1000,
                               fmt='%i', delimiter=',')
                    command = "regularized-k-means/build/regularized-k-means hard "+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + ".csv " + str(num_clusters) + \
                              " -a " + save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_assignments -o"+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_summary.txt -t 20"
                    # print(command)
                    os.system(command)
                    lbls = np.loadtxt(save_folder + "/subfeats_cls" + str(cls) + "_n" + str(
                            num_clusters) + "_hard_assignments.csv", delimiter=',')

                    print(Counter(lbls.reshape(-1).tolist()))

                    for j in range(num_clusters):
                        targets_subcls[subindex[lbls == j]] = subcls + j

    for i, file in enumerate(files):
        tgt_subcls = targets_subcls[file_id==i]
        feat_shape = feat_shape_list[i]
        tgt_subcls = tgt_subcls.reshape(feat_shape)
        H, W = label_shape_list[i]
        tgt_subcls = cv2.resize(tgt_subcls, (W,H), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(tgt_subcls).save(os.path.join(save_folder, file.replace('.npy','.png')))
    print(Counter(targets_subcls.reshape(-1).tolist()))

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,)
    parser.add_argument('-ca', '--clustering_algorithm', default='balanced_kmeans', type=str,
                        help="Support 'balanced_kmeans' or 'normal_kmeans'")
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config, args.clustering_algorithm)