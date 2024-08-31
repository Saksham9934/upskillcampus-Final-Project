# upskillcampus-Final-Project

**Skin Cancer Detection and Classification using YOLOv8 Model**

**Project Overview**

This project leverages the power of deep learning to detect and classify skin cancer using the state-of-the-art YOLOv8 model. Skin cancer is a growing concern worldwide, and early detection is crucial for effective treatment. Our project aims to develop an AI-powered tool that assists dermatologists in identifying skin cancer lesions from images, promoting timely interventions and improving patient outcomes.Skin cancer detection is a critical task in medical diagnostics, as early identification can significantly improve patient outcomes. In this study, we have take out the performance of advanced deep learning models—YOLOv8(You only look once) for the classification of skin lesions into benign and malignant categories using the ISIC 2020 dataset. YOLOv8 achieved an accuracy of 98.5%, with a precision of 0.98, recall of 1.00, and an F1 score of 0.99.

**Methodology**
We employed the YOLOv8 (You Only Look Once version 8) object detection model, renowned for its accuracy and speed. Our dataset consists of images of various skin lesions, annotated with their respective classes (benign or malignant). The model was trained on this dataset, enabling it to learn features and patterns characteristic of skin cancer.

**Key Features**
- Real-time detection: Our model can detect skin cancer lesions in real-time, making it suitable for clinical applications.
- High accuracy: YOLOv8's advanced architecture ensures precise detection and classification of skin cancer lesions.
- Multi-class classification: Our model can classify lesions into different types of skin cancer, providing valuable insights for diagnosis and treatment.

**Technical Details**
- Model: YOLOv8
- Framework: PyTorch
- Dataset: [The ISIC 2020 challenge dataset, a large collection of dermoscopic images labelled as benign or malignant, was used in this study. The dataset comprises thousands of images from multiple sources, annotated by dermatology experts. For our analysis, we used 23,126 images which comprised of two categories benign and malignant. Dataset also had a csv file containing various information of patient and whether the lesion is malignant or benign which further helped in better evaluation of model. ]
- Training parameters: [ models were trained using the ISIC 2020 dataset over 20 epochs, with a learning rate of 0.001 and a batch size of 32.. ]

**Usage**
1. Clone the repository and install the required dependencies.
2. Prepare your dataset and update the configuration files accordingly.
3. Run the training script to fine-tune the YOLOv8 model on your dataset.
4. Use the inference script to detect and classify skin cancer lesions in new images.

**Data Preprocessing**
 Resizing images to a uniform size, normalizing pixel values, and augmenting the data through techniques such as rotation, flipping, and zooming to enhance the model's ability to generalize is what which is known as data preprocessing. These steps were essential to ensure that the models could handle variations in image quality and lesion appearance. Here, images are resized to 256*256 pixels.

**Model Used**
1. YOLOv8: YOLOv8 is an advanced object detection model known for its speed and accuracy. It uses a single neural network to predict bounding boxes and class probabilities, making it suitable for real-time detection tasks. For this study, we adapted YOLOv8 for binary classification of skin lesions.

**Evaluation Metrics**
The dataset was divided into train, validation and test data. After training and validating model they are evaluated on the test data.
The models were evaluated using the following metrics:
•	Accuracy: The percentage of correctly classified images.
•	Precision: The proportion of positive identifications that were actually correct.
•	Recall: The proportion of actual positives that were identified correctly.
•	F1 Score: The harmonic mean of precision and recall, providing a balance between the two.
•	Support: The number of instances in each class.

**Quantitative Results**
YOLOv8: The YOLOv8 model achieved an accuracy of 98.5%, with a precision of 0.98, recall of 1.00, and an F1 score of 0.99. The support for malignant lesions was 6,507.

**Contributions**

We welcome contributions to improve the model's accuracy, expand the dataset, or enhance the user interface. Please submit your pull requests or issues to collaborate on this project.

Conclusion

**Summary**
This study demonstrates the performance of YOLOv8 model in skin cancer classification using ISIC 2020 dataset. Compared to other models, YOLOv8 achieves higher accuracy, improvement and F1 score, making it suitable for instant, highprecision skin detection. The findings highlight the potential of deep learning models to improve diagnostic accuracy and patient outcomes.

**Impact**
The successful application of YOLOv8 and CNN or other models, in this research demonstrates the evolution of deep learning in diagnosis. By providing fast, accurate, and functional skin cancer diagnosis, these models can empower physicians, enabling earlier intervention and better patient care.


License
