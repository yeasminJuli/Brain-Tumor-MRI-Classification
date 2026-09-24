ATTENTION WAVES - FINAL MANUSCRIPT PACKAGE

Use these TWO images from this folder:
01_Correctly_Classified_Attention_Weights.png
02_Misclassified_Attention_Weights.png

Both are 300-dpi PNG figures with the same 0-1 colour scale. The manuscript term is "attention weights". No training or code execution is needed to use the figures. The previous preview folders are not the final package.

WHERE TO ADD THEM
Place both figures in the explainability section under a subsection such as "Attention-weight analysis". Insert the correct-prediction figure first, followed by the misclassification figure. Replace Figure X and Figure Y below with the manuscript's actual figure numbers. These paragraphs and captions are proposed text; they have not been inserted into the manuscript.

FIGURE X CAPTION - CORRECT PREDICTIONS
Attention-weight distributions for correctly classified test examples from (a) glioma, (b) meningioma, (c) no-tumor, and (d) pituitary-tumor classes. Each input image is paired with the eight normalized weights used to aggregate the BiLSTM outputs into a context vector. Darker blue indicates a larger weight on a shared 0-1 scale. Values are rounded to three decimal places; a displayed zero may represent a small positive value. Sequence steps denote contextual feature groups, not anatomical locations or tumor boundaries.

FIGURE Y CAPTION - MISCLASSIFICATIONS
Attention-weight distributions for four misclassified test examples: (a) glioma predicted as meningioma, (b) meningioma predicted as no tumor, (c) meningioma predicted as glioma, and (d) pituitary tumor predicted as meningioma. These examples cover all four observed true-to-predicted class error pairs. The colour scale and interpretation match Figure X. Concentrated attention can occur in incorrect predictions and should not be interpreted as evidence of diagnostic correctness or tumor localization.

MANUSCRIPT TEXT - METHODS
To examine the attention module, we extracted its eight softmax-normalized weights for each test image using the trained model in evaluation mode. These weights form the context vector as c = sum_t(alpha_t h_t), where h_t is the BiLSTM output at sequence step t. Each weight is nonnegative, and the eight weights sum to one. The visualizations use a common colour scale from zero to one to permit comparison across examples. We selected the first correctly classified filename in alphabetical order within each class. For misclassifications, we selected the first filename in alphabetical order for each observed true-to-predicted class pair. Selection did not depend on the magnitude or appearance of the attention weights. This yielded four correct examples and four error examples; the latter cover all four error types among the seven misclassified test images. The plots show illustrative examples rather than class-wide average behaviour.

MANUSCRIPT TEXT - RESULTS AND INTERPRETATION
Figure X shows that the attention distribution varies across the displayed inputs. The glioma and pituitary-tumor examples place their largest weights on sequence step 4 (0.913 and 0.892, respectively). The meningioma example assigns its largest weight to step 5 (0.535), whereas the no-tumor example distributes most of its weight across steps 3-5 (0.459, 0.268, and 0.206). These observations describe the selected examples and do not establish a characteristic attention pattern for each diagnostic class.

Figure Y demonstrates that concentrated attention also occurs in incorrect predictions. For example, the glioma image predicted as meningioma places a weight of 0.914 on step 4, while the pituitary-tumor image predicted as meningioma places a weight of 0.892 on step 5. Thus, a dominant attention weight alone does not establish prediction correctness. The plots expose how the model weights its contextual outputs, but they do not establish the anatomical validity or causal importance of those outputs.

MANUSCRIPT TEXT - INTERPRETATION LIMIT
The attention weights quantify the relative weighting of contextual BiLSTM outputs when forming the 256-dimensional context vector. They are not pixel-level tumor probabilities or segmentation masks. In the original implementation, the direct reshape groups CNN features into eight sequence steps without an explicit spatial row or column permutation. Consequently, the weights are reported by sequence-step index and are not projected onto anatomical regions. The context vector itself was not plotted because the present analysis concerns the attention weights used to construct it.

PLAIN-LANGUAGE GUIDE
Left: the input image.
Eight boxes: the eight weights for that image.
White/light blue: low weight. Dark blue: high weight.
All eight exact weights total 1 (100%). Equal weighting would be 0.125 (12.5%) each.
The true and predicted labels show whether the prediction was correct.
The boxes do not identify the tumor's location.

EXACT SELECTED IMAGE NAMES
Correct (a): glioma/Te-glTr_0000.jpg
Correct (b): meningioma/Te-meTr_0000.jpg
Correct (c): notumor/Te-noTr_0000.jpg
Correct (d): pituitary/Te-piTr_0000.jpg
Misclassified (a): glioma/Te-gl_0259.jpg
Misclassified (b): meningioma/Te-meTr_0002.jpg
Misclassified (c): meningioma/Te-me_0266.jpg
Misclassified (d): pituitary/Te-piTr_0002.jpg

REPRODUCIBILITY
Selected_Images_and_Exact_Weights.csv contains the paths, labels, probabilities, and full-precision weights used in these figures.
regenerate_figures.py recreates these two PNGs from the existing extracted test weights, without retraining or changing the model.
Original checkpoint: training_outputs_proposed_model_orgnl/training_outputs/hybrid_cnn_bilstm_attention_final.pth
All-image extraction: ../original_attention_visualization/attention_weights_all_test_images.csv
Run metadata: ../original_attention_visualization/run_details.json

FINAL AGREED EXPLANATION FOR THE CORRECTLY CLASSIFIED FIGURE
The visualization shows the relative weights assigned to the eight BiLSTM outputs during context-vector construction. It exposes the attention module’s aggregation behaviour but does not identify tumor boundaries or explain the cause of individual classification errors.

Each BiLSTM output is a 256-dimensional contextual feature vector. The attention module combines these eight vectors through a weighted sum to form one context vector for classification. The coloured boxes show the weighting coefficients, rather than the feature values or their anatomical locations; darker blue indicates a larger weight. In the displayed examples, step 4 receives the greatest weight for glioma and pituitary tumor, step 5 for meningioma, and steps 3–5 collectively receive most of the weight for the no-tumor image. These observations describe the selected examples only.

Use this final agreed explanation for the correctly classified figure. The earlier misclassification discussion above is optional background and is not part of this agreed paragraph.
