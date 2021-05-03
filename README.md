# GAN Versus: Anime Faces

Below is our implementation on the Anime Face Dataset using **DCGAN** and **StyleGAN**.

## DCGAN

### 1.1 Image Synthesis
We tested our DCGAN model performance on two scales of the dataset. We trained the model based on 1) 50 epochs of the full dataset and 2) 100 epochs of the shrinked dataset (with 19000 images). Other parameters are hold the same: learning rate is 0.0002, batch size is 128, image size is 64×64×3, size of input noise is 100, adaptive learning rate β_1is 0.5. We see that the quality of generated images is better when we use the full dataset, suggesting that the abundance of data contributed to the models capturing image features more accurately. Meanwhile, the increasing number of epochs cannot compensate for the lack of training image, resulting in lower quality of result (see Fig.1).

Fig.1. Left: Training with 50 epochs on the full dataset, Right:Training with 100 epochs on the shrinked dataset

Full Dataset               |  19,000 images
:-------------------------:|:-------------------------:
![](https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/training_50epochs_full.png) | ![](https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/training_100epochs_19000.png)

By training with the shrinked dataset, the models did a fair job capturing the major attributes. In contrast, using the full dataset, the model not only learns those attributes, but also represents them more accurately and generates less artifacts (see Table.1).

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/DCGAN_Table.png" /></div>

### 1.2. Feature Exploration
As the author of DCGAN pointed out, the manipulation of the Z representation, ie. the input noise, yields smooth transformation of certain characteristics in the image. We explored this based on the model trained on the full dataset with 50 epochs. Here, we used a fixed input Z, which consists of 100 variables drawn from a standard Gaussian distribution, and manipulated the selected variables one at a time while holding the others constant, in order to observe the impact of such variables on the generated image.

We selected 10 input variables (ie. 1st, 10th, 20th…, 90th in the Z representation), and for each of them, we adjusted it by multiplying it with a manipulation factor K that ranges from -3 to 3. For example, Fig.2 illustrates the manipulation of the Z variable, and uses it as the input to generate an image. For this random input, scaling the 1st variable changes the face contour of the anime character, making the width of the face shrink as K increases. Scaling the 20th variable seems to impact the face orientation. Some Z variables appear to control multiple features, producing an effect of varying the facial expression. The 90th variable seems to impact features such as face orientation, eye position, and mouth shape, allowing the transition of facial expression and the overall appearance of the character.

Fig.2. Manipulation of Z representation changes in the image attributes. Each row shows the transformation of output images when we adjust that Z variable with the corresponding manipulation factor K. 

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/Z_manipulation.png" /></div>

However, we cannot conclude that there is an explicit mapping between the manipulation of a specific Z variable and the impact on image attributes. For example, the manipulation of the 1st variable does not always change the face contour. With another random input (see Fig.3.), the 1st variable does not have a strong influence on the face contour, and it seems to have a tiny effect on the hair style. Instead, the 60th variable now appears to have a significant effect on face shape. At the same time, the 70th variable now impacts the eye color, and we did not observe this effect in the previous example.
 
Fig.3. Another example Z representation manipulation

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/Z_manipulation2.png" /></div>

The difference in the effect of the variable at the same index position in the Z representation suggests that DCGAN is incapable of allowing specific-control of certain image attributes. However, DCGAN does reveal a potential of learning meaningful attributes and generating images with smoothly transiting features. Our results show the limitations of DCGAN in that the model cannot explicitly separate the features, adjusting a specific variable in Z representation sometimes lead to changes in multiple attributes and result in a global effect on the synthesized image.

### 1.3. Quality Evaluation
Given the limitations of computational power, we attempted to calculate FID in the training process with a smaller dataset consisting of 9000 images, and used InceptionV3 as the feature extractor. In this toy dataset, we trained on 5 epochs to get an idea of the training quality. We noticed that the FID score is always high and fluctuating. This is explainable because InceptionV3 was trained on a different dataset and might not capture the images features for the anime faces. We did not have enough time to implement a better suited CNN model for feature extraction, so we decided to comment on the results with visual inspection.
 
Fig.4. Plot of FID 

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/DCGAN_FID.png" /></div>

We observed that the loss of the generator and the discriminator display trends of declining in the training. However, both of them have significant fluctuations, and the loss of the generator is always greater than that of the discriminator. This implies that the quality of fake images is unstable and that the generator fails to perfectly simulate the real images. As a result, artifacts and lack of realness accompany this high generator loss. 

 
Fig.5. Loss of the Generator and Discriminator

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/DCGAN/DCGAN_Loss.png" /></div>

## StyleGAN

### 2.1 Image Synthesis
Our StyleGAN implementation involves selecting the first 19,000 images from our full dataset of 63,632 anime faces. We cloned NVIDIA StyleGAN GitHub and used some of the scripts as starter codes while editing only the critical lines.5 Our images were also resized, converted to Tensorflow records (tfrecords is required since StyleGAN uses TensorFlow) and pre-processed before training our model for 3,500 iterations. After 9 hours of training, we were able to produce a model with a FID of 39.4008 which is quite decent considering our limited hardware and time. Below are 8x8 grids of our images at genesis and completion:

  
	Fig. 6. Left: Our trained images at genesis. Right: Our trained images at completion (after 3,500 iterations)
	
At Genesis                 |  Completion
:-------------------------:|:-------------------------:
![](https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/StyleGAN/fakes000000_first.png) | !![](https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/StyleGAN/fakes003500_last.png)

As you can see the model images look quite pleasant, considering our model was trained on Google Colab Pro (1 GPU) for less than one day using only 19,000 images. The original NVIDIA’s StyleGAN paper used two main datasets for two different models: Flickr-Faces-HQ (FFHQ) and CelebA HQ, 70,000 1024x1024 and 200,000 with mixed resolution images, respectively.6 Additionally, our model used a resolution of 64x64 which was much lower than that of NVIDIA’s (1024x1024).

### 2.2 Feature Exploration
The beauty of StyleGAN is its controllability. The traditional GAN (Goodfellow et al.,2014), operated like a blackbox where random noises go in and an image gets generated. In this project, we explored two controlling methods of StyleGAN: Style Mixing and Truncation..

Style Mixing allows us to embed styles at different levels of our generative layers to control various features. Using this method, we were able to transfer style from raw images to our destination images. This method works quite well even on our small model with a shallow network of only a few layers.

At the coarse layer (4x4), we noticed changes in face shape, pose, hairstyle and mouth. These are the bigger, more noticeable features.

At the middle layer (8x8), we noticed changes in eye brows, eye color, nose and hair texture. These features are starting to head towards the subtle direction.

At the fine layer (64x64), we noticed slight changes in color scheme, sharper hairline and shading. These features are definitely more subtle compared to our coarse layer but not too far from our middle layer due to our shallow network.
 
Fig. 7. Styles from source B (top row) were combined with the full image of source A (left column).

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/StyleGAN/style-mixing2.png" /></div>

Truncation tricks involved in controlling the W with Ψ, using this equation:

<center> W_new=W_avg+Ψ(W-W_avg) </center>

By truncatingWwe were able to drastically change our generated images. At Ψ=0, faces converged to the “mean” face and looked the same as our generated images without truncation. However, when we apply negative scaling to styles, we got the corresponding opposite or “anti-face.” Lastly, at higher values of Ψ, gender, hair length, coloring are flipped.

Fig. 8. Truncation Trick

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/StyleGAN/truncation.png" /></div>

### 2.3 Quality Evaluation
StyleGAN performed quite well on a limited dataset with limited GPU and training time. We believe training the full dataset for multiple GPU days and with all layers (up to 1024x1024) will lead to much better results in both FID and resolution. The paper used FID as the main measurement of quality so we only measured the FID value for our model which was 39.4008.

Overall, our shallow model produced results inline with the StyleGAN paper. This indicates that StyleGAN is applicable to non-natural images like animated and cartoon characters. We find the truncation trick to be especially interesting since a slight tweak in Ψ can lead to very drastic changes. StyleGAN is quite powerful, even on our simple model.
 
Fig. 9. Newly generated images from our StyleGAN model

<div style="text-align:center"><img src="https://github.com/mthnguyener/GAN_Versus-Anime_Faces/blob/main/Results/StyleGAN/generated_images.png" /></div>

## Thank you!

## References
1.	Goodfellow, et al., Generative Adversarial Networks, https://arxiv.org/abs/1406.2661
2.	Churchill, Anime Face Dataset, Kaggle, https://www.kaggle.com/splcher/animefacedataset
3.	Radford, Metz and Chintala, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, https://arxiv.org/abs/1511.06434
4.	DCGAN Tutorial, https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
5.	NVLabs, StyleGAN — Official TensorFlow Implementation, GitHub, https://github.com/NVlabs/stylegan
6.	NVIDIA – Karras, Laine and Aila, A Style-Based Generator Architecture for Generative Adversarial Networks (2019), https://arxiv.org/abs/1812.04948
