# vit_cluster
# # Image clustering using pre-trained ViT
Use a Vision Transformer pre-trained through the DINO protocol to extract features from images. Reduce dimensionality and visualize using t-SNE and cluster the images with HDBSCAN. 
Can be run as-is or deployed using a docker image.
![cluster visualization](/assets/cluster_output.png)

The images can be overlaid on their embeddings to better understand the latent space.
![image overlay](/assets/image_overlay.png)
