from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from django.http import JsonResponse

def home(request):
    return render(request, 'index.html')
def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def services(request):
    return render(request, 'service.html')
def diseasePage(request):
    return render(request, 'disease.html')
def testimonial(request):
    return render(request, 'testimonial.html')

model_path = "C:/Users/RTC/Desktop/Ai course/best_model.h5"
model = load_model(model_path)

# Define function to preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Define dictionary mapping class indices to disease names
class_to_disease = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}


import os

# Get the directory of the current file (views.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the Excel file
excel_filename = 'medi plant.xlsx'
excel_path = os.path.join(current_directory, excel_filename)
df=pd.DataFrame(pd.read_excel(excel_path,sheet_name='Sheet2'))
df_sheet1 = pd.read_excel(excel_path, sheet_name='Sheet1')

@csrf_exempt
def predict_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get uploaded image
        uploaded_image = request.FILES['image']
        # Save the uploaded image temporarily
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        # Make prediction
        prediction = model.predict(processed_image)
        # Get the class label (adjust this based on your model)
        class_label = np.argmax(prediction[0])
        # Get disease name from class label
        disease_name = class_to_disease.get(class_label, "Unknown Disease")
        row = df.loc[class_label] 
        treatment = row.get('Treatment', 'Default Treatment')
        name = row.get('Disease', 'Default Name')
        fungicides = [row.get('Fungicide 1', ''), row.get('Fungicide 2', ''), row.get('Fungicide 3', '')]
        fungicides = [f if not pd.isna(f) else '' for f in fungicides]    
        links={}
        for fungicide in fungicides:
                row_sheet1 = df_sheet1[df_sheet1['Fungicide'] == fungicide]
                if not row_sheet1.empty:
                    # Extract links from Sheet1
                    link_columns = [col for col in row_sheet1.columns if col.startswith('Link')]
                    links[fungicide] = [row_sheet1.iloc[0][col] for col in link_columns if not pd.isnull(row_sheet1.iloc[0][col])]    
                    
        print(fungicides)
        print(links)
        # Delete the temporary image file
        os.remove(image_path)
        # Return the result
        return JsonResponse({'result': disease_name, 'treatment': treatment, 'fungicides': fungicides,'links': links})
    else:
        return JsonResponse({'error': 'No image provided'}, status=400)

