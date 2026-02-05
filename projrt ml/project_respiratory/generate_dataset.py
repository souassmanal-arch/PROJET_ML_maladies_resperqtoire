import pandas as pd
import numpy as np
import random

# Define categories
classes = ['Normal', 'Pneumonia', 'COVID']
genders = ['Male', 'Female']
binary_yes_no = ['Yes', 'No']

# Number of samples (keeping similar size to original)
n_samples = 300

data = []

for _ in range(n_samples):
    # Randomly assign a class
    disease = np.random.choice(classes, p=[0.4, 0.3, 0.3])
    
    # Generate features based on disease to make it somewhat realistic
    gender = np.random.choice(genders)
    smoker = np.random.choice(binary_yes_no)
    
    # Age: purely random 20-90
    age = np.random.randint(20, 90)
    
    if disease == 'Normal':
        fever = 'No'
        cough = np.random.choice(['Yes', 'No'], p=[0.1, 0.9])
        breath_difficulty = 'No'
        oxygen_saturation = np.random.randint(95, 100)
        heart_rate = np.random.randint(60, 100)
        respiratory_rate = np.random.randint(12, 20)
        
    elif disease == 'Pneumonia':
        fever = np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
        cough = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
        breath_difficulty = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
        oxygen_saturation = np.random.randint(85, 96)
        heart_rate = np.random.randint(80, 120)
        respiratory_rate = np.random.randint(20, 30)
        
    elif disease == 'COVID':
        fever = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
        cough = np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
        breath_difficulty = np.random.choice(['Yes', 'No'], p=[0.7, 0.3])
        oxygen_saturation = np.random.randint(80, 95)
        heart_rate = np.random.randint(90, 130)
        respiratory_rate = np.random.randint(22, 35)

    data.append([
        age, gender, smoker, fever, cough, breath_difficulty, 
        oxygen_saturation, heart_rate, respiratory_rate, disease
    ])

columns = [
    'age', 'gender', 'smoker', 'fever', 'cough', 'breath_difficulty',
    'oxygen_saturation', 'heart_rate', 'respiratory_rate', 'disease_class'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv('dataset.csv', index=False)
print("Dataset created successfully.")
