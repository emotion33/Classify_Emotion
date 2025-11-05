import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os


## Extract the features per song
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    return np.hstack([mfccs, tempo, centroid, zcr]) 

class_dirs = {
        'happy': 'happy/',
        'sad': 'sad/',
        'angry': 'angry/',
        'surprise': 'surprise/',
        'disgust': 'disgust/',
        'neutral': 'neutral/'
        }

def load_data(class_dirs):
    features, labels = [], []
    le = LabelEncoder()
    class_names = list(class_dirs.keys())
    le.fit(class_names)

    for class_name, dir_path in class_dirs.items()
        for file in os.listdir(dir_path):
            if file.endswith('.mp3'):
                feats = extract_features(os.path.join(happy_dir, file))
                features.append(feats)
                labels.append(class_name)


    X = np.array(features)
    y = le.transform(labels)
    return X, y, le

X, y, le = load_data(class_dirs)

print(X, y, le)

## Train the Support Vector Classifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

print("Trained on", X_train)
print("Will be tested on", X_test)
print("Accuracy: ",accuracy_score(y_test, clf.predict(X_test)))


## Classify new mp3s
def classify_song(file_path, clf,le):
    feats = extract_features(file_path)
    pred = clf.predict([feats])[0]
    return le.inverse_transform([pred_idx])[0]


for file in os.listdir('songs_to_classify/'):
    if file.endswith('.mp3'):
        result = classify_song(os.path.join('songs_to_classify',file),clf,le)
        print(file,">>",result)







