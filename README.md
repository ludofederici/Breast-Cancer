# Development of a Convolutional Neural Network Model for Assessing the Severity of Breast Cancer from Biopsy Images

## Introduction

Breast cancer remains the most common form of cancer in the United States, with more than 300,000 new cases expected in the U.S. in 2023. Despite advancements in medical technology, oncologists continue to face challenges in diagnosing and treating this disease, underscored by the heterogeneity of tumors, challenges of early detection, and inconsistent guidelines about mammogram eligibility. This paper introduces an artificial intelligence-based diagnostic tool to identify future high-risk breast cancer cases based on biopsy images, discusses the predictive power of this tool, and highlights the advantages and pitfalls of utilizing it in a clinical setting.

The algorithm is designed to enhance medical decision-making in determining the urgency of treatment for patients undergoing breast cancer screening. It focuses on a crucial allocation decision based on risk: identifying which patients require immediate surgery and chemotherapy, and which can be assigned to "watchful waiting," thereby avoiding unnecessary invasive procedures for lower-risk cancers. This decision is primarily made by pathologists who examine biopsy slides after a mammogram in the period following the biopsy. The decision-making process may, however, be made in conjunction with oncologists. 

The alternative to using this algorithm is the conventional interpretation of slides, where the pathologist studies biopsy slides visually to determine the type, structure, arrangement, and grade of the cancer cells. While the biopsy remains a critical tool in diagnosing breast cancer, it does not always provide a clear picture of how serious the cancer might be, and there is subjectivity in the process. 

Ultimately, the goal is to predict the future likelihood of breast cancer based on current biopsy results, making it a prediction problem. The goal is to empower pathologists with a greater ability to ascertain a patient’s risk of breast cancer through the use of the algorithm. With this knowledge, pathologists could more accurately refer patients who are likely to develop breast cancer for more aggressive treatment, significantly improving patient outcomes.

## Input and Sample

Our Nightingale dataset links 175,000 biopsy slides from 11,000 unique patients to cancer registry data on cancer stage, electronic health record data on the presence of metastasis, and Social Security data on mortality. Each observation in the dataset corresponds to a microscopy image from a breast biopsy specimen, collected at the Providence Cancer Institute (Portland, OR) between January 1st, 2010, and December 31st, 2020. The raw database (that we named ‘slide_biopsy_mapping_df’) has 52,262 slides, of which 3,910 are labeled with a metastatic cancer diagnosis.

As our code in the appendix shows, we conduct a data filtering operation on the raw dataset. The purpose of the mask we used is to exclude cases where the metastatic diagnosis was made prior to the biopsy date, ensuring that only post-biopsy diagnoses are considered. This prevents treatment pollution (as explained in further detail later), where the effects of a treatment administered to a subset of a population unintentionally spread to those in the control group or the rest of the population not receiving the treatment. After the mask is applied, the new subset of the dataset contains only the rows where the metastatic diagnosis date is equal to or later than the biopsy date, narrowing our dataset down to 51,487 images. 

![cancer_df](https://github.com/ludofederici/Breast-Cancer/assets/106605828/5efe3575-ea91-4259-adfd-16ac2cd96302)

This biopsy data paired with outcome labels allows us to train algorithms to identify patients at high risk of poor outcomes (metastatic cancer, death), and compare them to the pathologist’s initial staging decision. Ideally, we would have used the whole filtered dataset to train our algorithm; however, due to limited computational power, we decided to only train our model on the first 500 slides, which contained 51 diagnoses of metastatic cancer (positive cases). This final dataframe was subsequently split into training and validation sets using an 80/20 split in order to create a “hold-out set.”

The above process can be summarized using the following Sankey diagram:

![sankey](https://github.com/ludofederici/Breast-Cancer/assets/106605828/7a6d7e63-fa76-4c4a-8fef-cb21909c7062)

## Label

In health – like in many other complex social domains where AI is applied – the variables we have (Y) are rarely the exact variables we want (Y*). In our algorithm, our true underlying target Y* is whether or not a person actually gets breast cancer; however, this piece of information is not available to us. Instead, our literal variable Y – i.e. the closest label we can achieve to our Y* – is whether or not someone actually gets breast cancer and is diagnosed with breast cancer in our specific healthcare center.

This difference between measured variables and unmeasured true variables of interest creates scope for potential measurement bias in Y, such that Y = Y* + Δ. Firstly, measurement is subjective: since a diagnosis is not an objective assessment, but rather an opinion, it may be the case that patients are misdiagnosed or undiagnosed by physicians. Secondly, measurement is selective and event-based: since diagnostic test results reflect the decision to test, our Y is affected by the decision of a patient to get a routine check, or the frequency with which such a routine check happens. Hence, the potential for a diagnosis depends on a patient’s clinic visit, which also may be affected by one’s wealth and access to the healthcare center.

## Prediction Diagram

With all of these considered, the diagram for this prediction problem is as follows:

![pred](https://github.com/ludofederici/Breast-Cancer/assets/106605828/f242683d-c862-45e9-9b8c-b7e68393be38)

The decision point (X0) refers to the clinical decision-making moment, such as determining the cancer treatment plan (T) or assessing metastasis risk. At X0, the primary input variables, particularly the biopsy images and some patient information, were available. However, long-term outcomes like mortality were not available at this point and were used for retrospective analysis and model training.

## Data Preparation

The dataset comprised histopathological slide images represented by their unique identifiers (slide_id). The images were pre-processed using the preview_slide function, which generated thumbnail images from the slides. These thumbnails were resized to 224x224 pixels to maintain uniformity and converted to numpy arrays. Due to it being very computationally expensive and time-consuming, we were only able to do this for a dataset subset of 500 slides.

A TensorFlow Dataset object was created from these processed images and their corresponding binary labels indicating the presence or absence of metastasis (strict_metastatic_dx). The dataset was first shuffled for randomness, then split into training and testing sets with an 80-20 ratio (following the Pareto principle), ensuring diverse samples for both training and validation while enhancing its ability to generalize.

A significant challenge in our study was the imbalance in the dataset, where the majority of slide images were labeled as non-metastatic (only 51 positive samples out of 500). To mitigate this, we employed class weighting during the training process using scikit-learn's compute_class_weight function from the utils module. 

This approach assigns a higher weight to the underrepresented class, thereby amplifying its significance during model training. Balancing the dataset in this manner ensures that the model does not become biased towards the majority class and can learn effectively from both classes, leading to a more generalized and robust predictive performance.

## Model Architecture

Our study utilized a Convolutional Neural Network (CNN) built on top of the powerful Resnet-50 image classification model that we implemented using TensorFlow, a powerful open-source software library for machine learning applications. The complete code for the following algorithm can be found in the appendix, but this section contains a summary and the rationale of our approach.

In binary classification tasks, especially in medical diagnostics, setting the appropriate threshold for deciding class labels from probabilistic outputs is crucial. In our CNN, the final layer uses a sigmoid activation function that outputs a probability value ranging from 0 to 1, indicating the likelihood of the presence of metastasis. Typically, a threshold of 0.5 is used to classify outputs into binary labels. However, given the critical nature of accurate metastasis detection and the consequences of false negatives, we chose a threshold of 0.35 (which we determined through an iterative process), meaning even images with a lower probability of metastasis are identified as positive, thus reducing the chances of missing true positive cases.

We utilized the ResNet50 model, excluding the top layer, with weights initialized from a pre-downloaded file. The base model's layers were frozen to preserve learned features. Custom layers were added atop ResNet, including dense layers with ReLU activation and L2 regularization, and dropout layers for regularization to prevent overfitting. The model, compiled with Adam optimizer, binary cross-entropy loss, and accuracy metrics, was trained on TensorFlow datasets with class weighting and early stopping for improved generalization.

The model, with adjusted parameters (learning rate, batch size, probability threshold) for predictions, underwent fitting with training data and epochs, followed by prediction and evaluation using test data. In this architecture, we aimed to leverage ResNet-50's feature extraction capabilities while customizing for this specific prediction problem, striking a balance between transfer learning and task-specific hyperparameter tuning.

## Results

In the medical context of breast cancer prediction, the high Recall (Sensitivity) of 0.8235 and the ROC AUC score of 0.9235 in our study are particularly significant. In our study, the high recall of 0.8235 is particularly noteworthy as it implies our model successfully identifies a large proportion of actual breast cancer cases, which is crucial in a medical setting where missing a positive case could have dire consequences. The ability to detect most positive cases, even at the expense of some false positives, is vital since a high detection rate is often prioritized over reducing false positives as evidenced by the low number of False Negatives below in the confusion matrix.

The ROC AUC score, nearing 1, signifies the model's excellent discriminative power between cancerous and non-cancerous cases. A score of 0.9235 reflects its high accuracy in diagnosis, which is vital in a clinical setting. These metrics underscore the model's potential for practical application in breast cancer detection, highlighting its effectiveness and reliability in a medical environment.
Furthermore, our model’s effectiveness is validated by how it evidently outperformed the naive approach, surpassing it with an accuracy of 0.87 as opposed to 0.83. The naive approach typically predicts the most frequent class without considering the features in the data, serving as a baseline for comparison. In the context of breast cancer prediction, where the stakes are high, the improvement in accuracy, even if it appears modest, is significant.

## Pitfalls

Although we have just concluded that our prediction model shows potential, it is also important to recognize its pitfalls and implications.
 
Firstly, selective labeling complicates many prediction problems in medicine: the decision (e.g., whether or not to refer a patient to more aggressive cancer treatment) often reveals the label (e.g., whether or not the patient is at high risk of breast cancer/metastasis). In other words, the judgment of the decision-maker determines which instances are labeled in the data. As a result, there exist unmeasured variables that may be available to the decision-maker when making judgments but are not recorded in the dataset and hence cannot be leveraged by predictive models.

When observing the pathology labeling and diagnosing, factors other than the actual presence of cancer or metastasis could be present that may influence the pathologist in making their decision, such as biases in how pathologists interpret biopsy images. For instance, there are a number of molecular tumor features with prognostic value that may be observed differently, depending on the pathologist. These pathology entries may be approached subjectively because a surgical margin from one pathologist could be documented as “no residual tumor,” but another pathologist could document that same image as “microscopic residual tumor.”

It is crucial to observe the variable for metastatic diagnosis, as it is defined in a rather stringent method in the dataset provided. The issue is that the earliest diagnosis is used for each patient. In fact, the earliest metastatic diagnosis may predate a given biopsy as the dataset provided is representative of all the biopsies for breast cancer at the cancer center, and a biopsy may be involved in determining whether there is a recurrence or progression of cancer. That is, in order to label the patient with metastasis, both a breast cancer diagnosis and a metastatic diagnosis need to be present on the same day of the biopsy. If neither is present at the same time, then the patient will be labeled with no metastasis. Taking this into account, since our algorithm has a sample size of only 500, the total metastatic diagnosis was only 51.

As we navigate the mortality outcome of each patient, the dataset only includes whether there exists a death record or not in any of the three sources utilized–the cancer registry, Social Security Death Index, and the cancer center’s medical records–but the cause of death remains unknown. The dataset does not include the specification of “breast cancer mortality,” just “mortality.” Whether breast cancer or metastasis was the cause of death remains unknown.

Secondly, when looking at treatment pollution, an algorithm will take a strange lesson from the data in the training set: patients with the most obvious signs of high-risk cancer development are less likely to develop cancer because they’re promptly diagnosed and treated by doctors in the training dataset. The consequence of implementing such a system is that the high-risk patients are identified as low-risk and, thus, could have delayed diagnosis and treatment as a result.

Moreover, if a patient is diagnosed with high-risk breast cancer and receives treatment, it's challenging to observe the counterfactual. This treatment pollution affects the ability to see what the natural progression of the disease would have been, complicating the evaluation of the treatment’s effectiveness. To account for this treatment pollution, we filtered out patients who were diagnosed before the biopsy. We did so by fitting our algorithm on people who did not receive treatment, i.e., the pathologist did not identify the patient as high-risk. Additionally, of the treated people, we observed who developed cancer even though they received treatment, i.e., the treatment didn’t work.

Lastly, we had to take a smaller sample of our filtered dataset, reducing the size of our sample to 500. This was because, unfortunately, we do not have enough computing power to run the algorithm on the entire dataset. In addition to the large sample size, the environment didn’t allow us to import powerful image classification models like ResNet so we could not fine-tune it to fit our data. We also could not download data onto the local repository because it was too big (10s of TB). In a future research paper, it would be interesting to navigate around these restrictions and analyze how the accuracy of the algorithm changes.

## Conclusion

The introduction of the artificial intelligence-based diagnostic tool for identifying high-risk breast cancer cases represents a first step towards advancing the use of AI in the identification and treatment of breast cancer. This paper discussed the algorithm, its role in enhancing clinical decision-making processes, and its predictive capabilities. 

The value of this algorithm to clinicians is multifaceted, offering a new dimension in the management of breast cancer screening and treatment. The AI tool provides an objective analysis of biopsy images, reducing the subjectivity and variability inherent in conventional slide interpretations by pathologists. This consistency in diagnosis can lead to more accurate identification of cancer risk, ultimately improving treatment plans and patient outcomes. By assisting in the crucial decision of differentiating between cases requiring immediate intervention and less severe cases, the algorithm empowers clinicians to make more informed choices. This could improve  treatment plan selection, leading to better health outcomes for patients. The tool was promising in these goals, yielding an accuracy of 90.8% in the validation dataset, which exceeds the naive approach of predicting no breast cancer for all patients but needs to be benchmarked against the accuracy of pathologists. However, substantial additional work would need to be done prior to deploying an approach like this in a clinical setting, including benchmarking against pathologists and validation in novel datasets. 

If additional benchmarking and success in novel datasets can provide further evidence of the value of this algorithm, hospitals/insurers would recognize the outcome improvement and total treatment cost reductions possible and could be motivated to pay for it. They might further realize that after the algorithm is trained and deployed, the marginal cost of using the algorithm on existing biopsy slides may be very low. A diverse set of stakeholders might be interested in paying for this: hospitals who want better resource allocation, patients who want better health, and payers who want to reduce per-patient spending through earlier detection.

In conclusion, the AI-based diagnostic tool offers promising advancements in the diagnosis and management of breast cancer. Its potential to augment clinical decision-making, reduce healthcare burdens, and improve patient outcomes is considerable. However, continuous development, validation, and integration into clinical workflows are essential for realizing its full potential. As the tool evolves, it could significantly transform the landscape of breast cancer care, ushering in an era of more precise, personalized, and proactive oncology.

![diagrams](https://github.com/ludofederici/Breast-Cancer/assets/106605828/7347cc03-c223-4d6e-b430-400a397f3f02)

![matrix](https://github.com/ludofederici/Breast-Cancer/assets/106605828/a2d90c41-8f7c-4233-bbfa-0da85b69cd25)
