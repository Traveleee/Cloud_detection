For the problems such as attention noise interference, loss of detailed features, and limited generalization ability in the cloud detection task of small-
sized multispectral remote sensing images, this paper proposes an improved cloud detection algorithm that integrates the differential attention mechanism and 
the U-Net architecture. This algorithm introduces a differential Transformer module with noise suppression characteristics, eliminates the common-mode noise 
interference in the attention mechanism through dual-path Softmax differential operations, and constructs various data enhancement strategies to enhance the 
model's robustness against complex scenarios such as atmospheric scattering and thin cloud interference. Experimental verification of this model shows that 
the overall accuracy (94.46%), precision (83.31%), and F1-score (90.45%) of this algorithm are superior to the optimal ViT-UNet of the same type (overall 
accuracy: 92.37%, precision: 78.96%, F1-score: 83.74%). In the unseen data of the test set, the recall rate of Diff-UNet (87.21%) is also higher than that of 
ViT-UNet (78.07%), thereby verifying that this algorithm effectively reduces the probability of cloud detection misjudgment in small-sized training.
