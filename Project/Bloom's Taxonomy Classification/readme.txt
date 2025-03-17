AI Cognitive Classification

📌 Project Overview

AI Cognitive Classification is a machine learning-based solution designed to classify educational questions according to Bloom’s Taxonomy, an essential framework in education for categorizing learning objectives. This project employs Word2Vec embeddings, TF-IDF vectorization, and a variety of machine learning algorithms to automate question classification with high accuracy and efficiency.

📜 Table of Contents

Project Overview

Key Features

Technologies Used

Installation

Usage

Dataset

Results & Performance

Future Work

Contributors

License

🎯 Key Features

Automated Question Classification: Uses ML models to classify questions based on Bloom’s Taxonomy cognitive levels.

Diverse ML Models: Implements SVM (Linear & RBF), Random Forest, Decision Tree, Logistic Regression, and KNN.

Advanced Preprocessing: Cleans and vectorizes text data using Word2Vec embeddings and TF-IDF.

High Accuracy: Achieves up to 99.69% accuracy, surpassing prior research benchmarks.

Comprehensive Evaluation: Uses confusion matrices, ROC curves, precision-recall curves, and Cohen’s Kappa scores.

💡 Technologies Used

Programming Language: Python 🐍

Libraries & Frameworks:

Scikit-learn (Machine Learning Models)

NLTK & SpaCy (Text Preprocessing)

Pandas & NumPy (Data Handling)

Matplotlib & Seaborn (Visualization)

Data Processing: Word2Vec, TF-IDF

Machine Learning Models: SVM, Random Forest, Decision Tree, Logistic Regression, KNN

🛠 Installation

To set up the project locally, follow these steps:

Clone the repository:

git clone https://github.com/your-repo-link.git
cd ai-cognitive-classification

Create and activate a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Download and preprocess the dataset:

python preprocess.py

Run the model training:

python train_model.py

🚀 Usage

Once the setup is complete, you can use the trained model to classify educational questions:

python classify.py --input "What is the capital of France?"

The output will return the Bloom’s Taxonomy category for the question.

📊 Dataset

Size: 6458 questions

Sources:

Academic databases

ChatGPT-generated augmentations

Publicly available question banks

Categories: Analyze, Apply, Create, Evaluate, Remember, Understand

📈 Results & Performance

Best Performing Model: SVM (RBF Kernel) – 99.69% Accuracy

Evaluation Metrics:

Accuracy

Precision-Recall Scores

Confusion Matrices

ROC Curves

Cohen’s Kappa Agreement

Example accuracy comparison:

Model

Accuracy

SVM (RBF Kernel)

99.69%

SVM (Linear)

98.45%

Random Forest

97.20%

KNN

94.65%

Decision Tree

91.30%

Logistic Regression

96.88%

🔮 Future Work

Multi-Category Classification: Enhancing classification for multi-label question tagging.

Ensemble Methods: Improving accuracy using combined models.

User Feedback Loop: Incorporating real-time feedback for model improvement.

Deployment: Developing a web-based or API-based system for easy accessibility.

👥 Contributors

Gaurav (@Dx2905)

Hao Sheng Ning

Linjing Xu

Seyed Mohammad Ghavami

📍 Khoury College, Northeastern University, Portland, Maine, USA

For inquiries, contact: lnu.gau@northeastern.edu

📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

🚀 If you find this project helpful, give it a ⭐ on GitHub! 🎉

