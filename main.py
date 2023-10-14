#%% 
# Import libraries and helper functions
import os
import argparse
import torch
import datetime

import torch.nn as nn
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
#from sklearn.cluster import HDBSCAN
import csv

#import umap
#%%
from datasets import ImageDataset, Dataset, bbox_iou
#from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
from object_discovery import ncut 
import matplotlib.pyplot as plt
import time
from torchvision import transforms as pth_transforms
import glob
from torch.utils.data import DataLoader
import cv2
from openTSNE import TSNE
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import dino.vision_transformer as vits
#%%
# Pass in the path to the image directory

parser = argparse.ArgumentParser("Visualize Self-Attention maps")
parser.add_argument("--path",type=str,help="path to dataset")
parser.add_argument("--output",type=str,help="name of output file")



class MarineBenthicDataset(Dataset):
    """seabed dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        
        image_filepath = self.root_dir[idx]
        image_name = os.path.basename(image_filepath)
        
        image = mpimg.imread(image_filepath)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        sample = {'image': image, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensorCustom(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, name = sample['image'], sample['name']
                         
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        # Applying a normalization to pixels according to this forum: https://discuss.pytorch.org/t/understanding-transform-normalize/21730/14
        # This makes each channel zero-mean and std-dev = 1
        norm_transform = pth_transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        image =norm_transform(image)
        return {'image': image,'name': name}

class RescaleCustom(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, name = sample['image'],sample['name']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        

        return {'image': img, 'name': name}    

def load_dataset_dataloader(directory_path):
    # Empty list to store paths to images
    image_paths = []
    
    
    # Grab paths to images in train folder
    for data_path in glob.glob(directory_path + '/*'):
        
        image_paths.append(data_path)
        
    image_paths = list(image_paths)
    transform = pth_transforms.Compose([RescaleCustom(64),ToTensorCustom()])
    #transform = pth_transforms.Compose([ToTensorCustom()])
    # create datasets for training, validation and testing. 
    dataset = MarineBenthicDataset(image_paths,transform=transform)
    # load training data in batches
    data_loader = DataLoader(dataset, 
                              batch_size=32,
                              shuffle=True, 
                              num_workers=0)
    
   
    return dataset,data_loader,image_paths


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, plot_size=100, max_image_size=200):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    max_image_size=150
    plot_size = 3000 #1000
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset
    
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    
    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, x, y in zip(images, tx, ty):
        image = cv2.imread(image_path)
    
        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)
    
      
    
        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
    
        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
    tsne_img_plot = plt.figure(figsize=(25,25)) # plt.figure(figsize=(15,50))
    plt.imshow(tsne_plot[:, :, ::-1])
    plt.grid()
    plt.show(block=False)
    return tsne_img_plot


#%% 

# Choose whether ground truth labels are available or not
ground_truth_available = False


#%% 
# Create datasets and dataloaders

# Specify path to dataset directory
args = parser.parse_args()

path_to_dir = args.path

path_to_dir_reference = '/media/surajb/suraj_drive/datasets-acfr/lizard_island_03_22/images_trimodal/'
path_to_dir_target = '/media/surajb/suraj_drive/datasets-acfr/lizard_island_03_22/i20220403_000804_cv_northreef/'
#path_to_dir = 'resized_output'

dataset,dataloader,image_paths = load_dataset_dataloader(path_to_dir)

# Load model
patch_size=16
# Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device('cuda') 

# Specify constants for ViT-small architecture
arch="vit_small"
features_per_patch=383
patch_size = 16

# Load model and send it to gpu 
model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

for p in model.parameters():
        p.requires_grad = False

model.load_state_dict(torch.load('dino_deitsmall16_pretrain.pth'))
model.eval()
model.to(device)



#%% Forward pass images
# Loop over images
preds_dict = {}
output_dict = {}
cnt = 0
#cls_token_array = np.zeros((len(dataset),384))
#corloc = np.zeros(len(dataset.dataloader))
#cls_token_dataframe = pd.DataFrame()
cls_list = []
eigenval_list = []
cost_list = []
im_name_list = []
second_eigen_vector = []
bbox_area_list = []
attn_conc_list = []
rescaled_eigenvector_list = []
start_time = time.time() 
pbar = tqdm(dataset)
with open(str(args.output),'a') as f1: # with open('test_csv.csv','a') as f1: 
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp['image']

        init_image_size = img.shape

        # Get the name of the image
        im_name = inp['name']
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.floor(img.shape[1] / patch_size) * patch_size),
            int(np.floor(img.shape[2] / patch_size) * patch_size),
        )

        img = img[:,:,:size_im[2]]
    
        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        
        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            
            # Store the outputs of qkv layer from the last attention layer
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            # Forward pass in the model
            attentions = model.get_last_selfattention(img[None, :, :, :]) # attentions = model.get_last_selfattention(img[None, :, :, :])

            # Scaling factor
            scales = [patch_size, patch_size]

            # Dimensions
            nb_im = attentions.shape[0]  # Batch size
            nh = attentions.shape[1]  # Number of heads
            nb_tokens = attentions.shape[2]  # Number of tokens

            # Baseline: compute DINO segmentation technique proposed in the DINO paper
            # and select the biggest component
            
            # Extract the qkv features of the last attention layer
            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            feats = k
    

        # ------------ Apply TokenCut ------------------------------------------- 
        
        #pred, objects, foreground, seed , bins, eigenvector,all_eigenvectors,eigenvals,bipartition,cls_token,cost,bbox_area,_,_ = ncut(feats, [w_featmap, h_featmap], scales, init_image_size, tau=0.2, eps=1e-5, im_name=im_name, no_binary_graph=False)
        pred, objects, foreground, seed , bins, eigenvector,all_eigenvectors,bipartition,cls_token,rescaled_bipartition,rescaled_eigenvector = ncut(feats, [w_featmap, h_featmap], scales, init_image_size, tau=0.1, eps=1e-5, im_name=im_name, no_binary_graph=False)  
        
        #diff_attn = np.square(rescaled_eigenvector_list[610]-rescaled_eigenvector_list[611])
    
        #cls_list.append(np.squeeze(cls_token))
        
        
        writer=csv.writer(f1)
            
        row =np.append(args.path+im_name,np.array(cls_token)) 
        writer.writerow(row)

        #im_name_list.append(im_name)
        
        cnt+=1

        #cls_token_array[im_id,:]=cls_token
    end_time = time.time()
print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')


#%%
# Place data in dataframes
#cls tokens
cls_data = pd.DataFrame(cls_list)


# Second eigen vector for each image
second_eigen_data = pd.DataFrame(second_eigen_vector)




# Attention-concentration dataframe
attn_conc_dataframe = pd.DataFrame(attn_conc_list)

# Add image name column to data of interest
cls_data['image_name'] = im_name_list
#cls_dataframe = cls_data.set_index('image_name')
cls_dataframe = cls_data
second_eigen_data['image_name'] = im_name_list


# write cls_data to csv file (store data in case of crash)
cls_dataframe.to_csv('crawler_features.csv')



#%% 
if ground_truth_available:
    ground_truth_data = pd.read_csv('timestamped_feature_output.csv',usecols=['file_name','ground_truth_label','time','local_north','local_east','depth'])
    left_df = cls_dataframe
    right_df = ground_truth_data
    merged_dataframe = pd.merge(left_df, right_df,how='inner',left_on="image_name",right_on="file_name")
    # Extract dataframe of target ground truth
    #target indexes
    target_index = merged_dataframe[merged_dataframe['ground_truth_label']==1].index
else:
     merged_dataframe = cls_dataframe  

#%%

# Sort merged dataframe in ascending order of time
sorted_merged_dataframe = merged_dataframe.sort_values(by=['time'],ascending=True)
#%%
time_sorted_eigen_vectors = [rescaled_eigenvector_list[i] for i in sorted_merged_dataframe.index.values]

#%% Reduce feature dimensions using tsne


cls_token_array = MinMaxScaler().fit_transform(merged_dataframe.iloc[:,0:features_per_patch])

tsne = TSNE(n_components=2,
    perplexity=30,
    metric="cosine",
    n_jobs=8,
    random_state=42,
    verbose=True,initial_momentum=0.5,final_momentum=0.8,dof=0.2
)
'''
umap_reducer = umap.UMAP(n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,densmap=True,)
'''  
embedding = tsne.fit(cls_token_array)
#embedding = umap_reducer.fit_transform(cls_token_array)
# Normalize embedding
embedding_norm = MinMaxScaler().fit_transform(embedding)


#%% 
# Save outputs

output_dataframe = pd.DataFrame()
output_dataframe['image_name'] = im_name_list
#output_dataframe['density_Scores'] = prob_density_scores_norm
#output_dataframe['prediction'] = (prob_density_scores_norm>prob_density_threshold)*1
output_dataframe['embedding_norm_0'] = embedding_norm[:,0]
output_dataframe['embedding_norm_1'] = embedding_norm[:,1]
output_dataframe.to_csv('outputs/output_file.csv')


#%% Fit a kde to the latent data 

from sklearn.model_selection import GridSearchCV
from scipy import stats

# Define a grid
import math
num_points = 100

x = np.linspace(0, 1, num_points)
y = np.linspace(0, 1, num_points)

xv, yv = np.meshgrid(x, y)

xy = np.vstack([yv.ravel(), xv.ravel()]).T

density_model = stats.gaussian_kde(embedding_norm[:,:2].T)
zv = np.reshape(density_model.evaluate(xy.T).T, xv.shape) #density_model.evaluate

levels = np.linspace(0, zv.max(), 50)




#%% 
# Calculate density scores
prob_density_scores = density_model.evaluate(embedding_norm[:,:2].T)

norm_factor = np.sum(prob_density_scores)
prob_density_scores_norm = -prob_density_scores #1-prob_density_scores/norm_factor
# log_pdf_scores = np.exp(-log_pdf_scores)
percentile_thresh = 0.99
prob_density_threshold = np.quantile(prob_density_scores_norm, percentile_thresh)
print(prob_density_threshold)
b_width=0.05 #was originally 0.00001


fig,ax = plt.subplots(2,figsize=(15,20))
ax[0].set_xlim([0,1.01])
ax[0].set_ylim([0,1.01])
#ax[0].contourf(yv,xv,zv,levels=levels,cmap='Blues',alpha=0.8)
ax[0].scatter(embedding_norm[:,0],embedding_norm[:,1],alpha=0.2,edgecolors='none')
ax[0].grid()

ax[1].hist(prob_density_scores_norm,density=True, alpha=0.5, bins=np.arange(min(prob_density_scores_norm), max(prob_density_scores_norm) + b_width, b_width),label='Inliers')
ax[1].axvline(x=prob_density_threshold,color='k',linestyle='dashdot')
ax[1].grid()
if ground_truth_available:
    ax[0].scatter(embedding_norm[np.array(target_index),0],embedding_norm[np.array(target_index),1],c='red',alpha=0.2)
    ax[1].hist(prob_density_scores_norm[np.array(target_index)],color='r',density=True, alpha=0.5, bins=np.arange(min(prob_density_scores_norm), max(prob_density_scores_norm) + b_width, b_width),label='Outliers')
    ax[1].legend()

plt.savefig('outputs/output_cluster.png')

#%% Overlay images on latent space
# Specify proportion of images to overlay
num_overlay = int(0.5*len(dataset))
random_indexes = np.random.randint(0,len(image_paths),num_overlay)

tsne_array_random_subset = np.array(embedding_norm[random_indexes])
images_random_subset = np.array(image_paths)[random_indexes]



image_overlay = visualize_tsne_images(tsne_array_random_subset[:,0],tsne_array_random_subset[:,1],images_random_subset)

image_overlay.savefig('outputs/image_overlay.png')




#%% 

 

X = embedding_norm


# Try DBSCAN

from scipy import stats
from sklearn.cluster import DBSCAN,HDBSCAN
from sklearn import metrics



#db = DBSCAN(eps=0.025, min_samples=6).fit(X[:,:2])


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 10))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.jet(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="None",
                markersize=6 if k == -1 else 1 + 10 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.grid()
    #plt.tight_layout()

db = HDBSCAN(min_cluster_size=5).fit(X[:,:2])

plot(X, db.labels_, db.probabilities_)
plt.savefig('outputs/cluster_output.png')
plt.show()

labels = db.labels_


#%%

labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))



plt.title("Estimated number of clusters: %d" % n_clusters_)





#%% 

# Display a random image from a chosen cluster
#image_dir = '/media/surajb/suraj_drive/flinders_crawler_rosbags/crawler_hmi_2023_07_17_02_07_50/output_images/'
#image_dir = '/media/surajb/suraj_drive/datasets-acfr/lizard_island_03_22/images/'
image_dir = path_to_dir
# save images to disk

# Add labels to dataframe

output_dataframe['cluster_labels'] = labels

num_unique_labels = output_dataframe['cluster_labels'].nunique()
#print(num_unique_labels)

unique_cluster_labels = output_dataframe['cluster_labels'].unique()
#print(unique_cluster_labels)

# Number of samples to plot per cluster
num_samples_per_cluster  = 6

for index,label in enumerate(unique_cluster_labels):
    fig,axs = plt.subplots(1,num_samples_per_cluster,figsize=(15,5),constrained_layout=True)
    subset_data = output_dataframe.loc[output_dataframe['cluster_labels']==label]
    samples = subset_data['image_name'].sample(num_samples_per_cluster, replace=True)
    image_plot_path =image_dir+samples
    for i in range(num_samples_per_cluster):
        axs[i].imshow(plt.imread(image_plot_path.iloc[i]))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig('outputs/'+str(label)+'.png')

output_dataframe.to_csv('outputs/cluster_labels.csv')