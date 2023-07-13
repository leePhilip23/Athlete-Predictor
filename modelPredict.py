import numpy as np
import cv2
import os
import shutil
import joblib 
import pickle
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# Wavelet Transform to get specific image for comparison
def w2d(img, mode='haar', level=1):
    imArray = img
    # Convert Datatype
    # Convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # Compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
  

# Preprocessing: Load image, detect face. If eyes >= 2, then save and crop the face region
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color



img = cv2.imread('./test_images/sharapova1.jpg')

# Make color gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Change face and eyes into numerical values
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h) = faces[0]
face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


# Draw box shapes around face and eyes
cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

# Show face once more more zoomed up
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()


# Turn cropped image to numerical value
cropped_img = np.array(roi_color)
  
  
# Show wavelet transform
im_har = w2d(cropped_img,'db1',5)
plt.imshow(im_har, cmap='gray')


# Test to see if the image of having two visible eyes work
cropped_image = get_cropped_image_if_2_eyes('./test_images/tiger_woods.jpg')
plt.imshow(cropped_image)

# Test to see if it works on image 2 (It shouldn't since 2 eyes aren't visisble)
cropped_image_no_2_eyes = get_cropped_image_if_2_eyes('./test_images/sharapova2.jpg')
cropped_image_no_2_eyes

# Path to Data
path_to_data = "./dataset/"
path_to_cr_data = "./dataset/cropped/"


# Import dataset (Pictures)
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

# Make sure the datasets are valid
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)


# Automatically crop every image
cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1


# Examine cropped folder and delete any unwanted images
celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list
celebrity_file_names_dict


# Storing names in dictionary
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict


# Images in cropped folder -> model training
X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name]) 

        
# Transfrom into 2D array
X = np.array(X).reshape(len(X),4096).astype(float)



# Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# Print classification report
print(classification_report(y_test, pipe.predict(X_test)))


'''
# Uses svm, random forest, or logistic regression depending on what part of dictionary is called
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

'''
# Filters are kernels/nodes in first layer, kernel_size -> 3x3, padding -> max pooling for every value
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[256, 256, 3]))

# First convolutionary and max pooling layers
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
# Dropout Layer
model.add(tf.keras.layers.Dropout(0.5))


# Second convolutionary and max pooling layers
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

model.summary()

# ResNet50 (transfer learning) Architecture Below
resNet50 = tf.keras.models.Sequential()
transfer = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights=None,
                input_shape=(256,256,3),
                pooling='max',
                classes=4
            )

resNet50.add(transfer)
resNet50.add(tf.keras.layers.Flatten())
resNet50.add(tf.keras.layers.Dense(units=128, activation='relu'))
resNet50.add(tf.keras.layers.Dense(units=4, activation='softmax'))

resNet50.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resNet50.fit(X_train, y_train, epochs=100)

# Makes Predictions
scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])


best_estimators['svm'].score(X_test,y_test)
best_estimators['random_forest'].score(X_test,y_test)
best_estimators['logistic_regression'].score(X_test,y_test)

# Find accuracy of resnet-50
y_predict = []
ROW = len(y_pred)
COL = len(y_pred[0])
for i in range(ROW):
    maximum = -1
    index = -1
    for j in range(COL):
        if y_pred[i][j] > maximum:
            maximum = y_pred[i][j]
            index = j
    y_predict.append(index)

y_predict = np.array(y_predict)
print(classification_report(y_test, y_predict))


# Pickle File
pickle.dump(best_clf, 'saved_model.pkl') 
pickle.dump(resNet50, 'cnn.pkl')
