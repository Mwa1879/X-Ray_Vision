# X-Ray_Vision (FULL REASEACH IS IN THE PDF)
The number of chest x-rays performed each year has
been increasing and each requires an expert radiologist to
manually analyze and provide a diagnosis. In order to assist
these physicians and tackle the problem of doctor error, we
designed a CNN to classify common pulmonary diseases
as well as COVID-19. In this project, we also compare our
model to DenseNet121 and MobileNet for the task of chest
x-ray image classification. We find that although our model
has slighly higher binary accuracy, both MobileNet and
DenseNet121 have greater AUC-ROC values. Our project
integrated elements from a kaggle notebook that we found
here. Code and dataset are available at this Githuhb Repo
here.
# 1. Introduction and Motivation
Chest X-Rays are one of the most commonly performed
radiological examinations for screening and diagnosis of
many lung diseases. According to the WHO, each year, about
4 million people die from lower respiratory infections and
pneumonia [5]. Within healthcare systems, radiologists manually classify several hundreds of x-rays each day without
any automated assistance. Furthermore, the retrospective
error rate among radiologic examinations is approximately
30%, with real-time errors in daily radiology practice averaging 3–5%. A study from the Mayo Clinic found that when
seeking a second opinion, the original diagnosis was only
confirmed 12% of the time. Conversely, as many as 88% of
patients are given a new or refined diagnosis [9]. Diagnostic
errors are estimated to account for 40,000–80,000 deaths
annually in U.S. hospitals alone. [4]
In addition, during the COVID-19 pandemic, hospitals
in the US and around the world have been overwhelmed
by the number of patients seeking admission for respiratory
illnesses. COVID-19 predominantly targets the lungs and is a
highly infectious disease with an estimated mortality rate of
1.3% [2]. The early and automatic diagnosis of COVID-19
may be beneficial for timely recommendation of the patient
to quarantine, rapid intubation of serious cases in hospitals,
and monitoring of the spread of the disease.
# 2. Problem Statement
Our goal was to build a convolutional neural network
model to assist radiologists with correctly classifying different common pulmonary conditions, such as pneumonia,
atelectasis, hernia, edema. This would help to reduce the
immense work burden on doctors, especially because the
number of x-rays performed is constantly increasing, and
allow for patients to get more accurate diagnosis. Furthermore, we wanted to use COVID-19 chest x-rays to see if
our model could accurately classify COVID-19 x-rays. This
would allow for greater efficiency in diagnosing COVID-19
cases, which would enable hospitals to provide better patient
care and help more people.

# 3. Related Work
Deep learning has been applied to many fields within
medicine. Some noteworthy achievements in this field have
been the use of CNNs to achieve dermatologist-level classification of skin cancer [3] and radiologist-level pneumonia
detection on chest x-rays [7]. As more complex neural network models are being developed, the accuracy of these deep
learning classification systems have been increasing. Furthermore, the medical research community has come together to
release large medical information datasets for purposes such
as deep learning.
In 2017, the NIH released a chest x-ray image dataset with
112,120 frontal-view x-ray images of 30,805 unique patients
with fourteen text-mined disease image labels [10]. Recently,
a Kaggle dataset containing about 190 COVID-19 x-ray
images was released. In this project, we combined these two
image datasets and built our own CNN model to classify
these images. We also fed our data through the existing
DenseNet121 and MobileNet models, and then compared
the results with the model we implemented.
1
# 4. Methods
Before we could use deep learning to classify our image
data, we had to perform extensive preprocessing in order
to rescale the images, convert them to gray-scale, generate
the image data, and create dictionaries for the labels. The
images in the NIH chest x-ray dataset were 1024x1024 pixels; however, the COVID-19 images were variable in their
sizes. We resized all of our images to a standard 256x256
pixels, which also helped to reduce the computational time
due to the size of our dataset. Furthermore, we chose to use
gray-scale rather than color images for our model because
was previously shown that models pre-trained on grayscale
ImageNet outperformed color ImageNet models for disease
classification based on chest x-ray images [11]. Reducing
our images to only 1 channel as compared to 3 channels
also improved our training speed. Lastly, for image data augmentation within our preprocessing pipeline, we chose to
randomly flip our images horizontally as well as add a slight
height shift, width shift, rotation, sheer and zoom.
The architecture of the CNN model that we built is shown
in Figure 1. We chose this design based on a combination
of three previous CNN architectures that had been shown
to be relatively successful in classifying x-ray images [8]
[6] [1]. Our model used the Adam optimizer and binary
cross entropy loss, and we tracked the performance using the
binary accuracy and mean absolute error metrics. In regards
to hyperparameters, our model used a batch size of 32 with
a learning rate of 0.001 and ran for 7 epochs. Once learning
stagnates, the learning rate is reduced by a factor of 3 in
an attempt to improve model performance. Regularization
was also implemented with a 50% dropout for each epoch
to prevent overfitting. The final layer of our model was an
multilayer perceptron with 16 units and a sigmoid activation
function to assign probabilities to each of the 16 disease
classes for a given image.
In addition to this model, we also included modified versions of the DenseNet121 and MobileNet, whose architectures are shown in Figure 2. These models contained few
parameters (all trainable) than our model; however, their
designs are much more complex. Our model was simpler in
order to achieve computational efficiency.
To run these different CNN models, we split our data up
into 75% for training and validation and 25% for testing.
Based on our initial results, we found a strong imbalance
within our dataset and reduced accuracy due to multiclass
images. To resolve this, we excluded images with the label
“No Findings,” which accounted for more than half of our
dataset. We also removed x-ray images which were under
a duplicate Patient ID and image label to reduce any skew
that may have been present. Additionally, because some
images were under several labels, we excluded these from
our dataset; however, in the future, we hope to adapt our
model for multiclass predictions


# Team contributions
Mohamad Abouelafia Worked on research, building the model/preprocessing, and training and testing the model.
Isaac Nathoo Worked on research, building the model/experimented with preprocessing, and writing the report.
Hossam Zaki Worked on research, building the model/preprocessing, and training and testing the model.
Muhammad Haider Asif Worked on research, building the model/experimented with preprocessing, and writing the report.
