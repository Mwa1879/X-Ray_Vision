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

# Team contributions
Hossam Zaki Worked on research, building the model/preprocessing, and training and testing the model
Isaac Nathoo Worked on research, building the model/experimented with preprocessing, and writing the report
Muhammad Haider Asif Worked on research, building the model/experimented with preprocessing, and writing the report
Mohammad Abouelafia Worked on research, building the model/preprocessing, and training and testing the model
