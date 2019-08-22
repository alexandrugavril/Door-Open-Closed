import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

matplotlib.use('tkagg')

URL = "rtsp://admin:admin@192.168.0.127/play4.sdp"

training_closed = ['./Door/Closed/door_closed3.avi', './Door/Open/door_closed2.avi']
training_open = ['./Door/Open/door_open.avi', './Door/Open/door_open2.avi']
test_closed = ['./Door/Open/door_closed4.avi', './Door/Open/door_closed.avi']
test_open = ['./Door/Open/door_open3.avi', './Door/Open/door_open4.avi']


def get_data(video, is_open=True):
    frames = []
    labels = []


    cap = cv2.VideoCapture(video)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            # cv2.rectangle(frame, (355, 250), (390, 275), (0, 255, 0), 3)
            crop_img = frame[250:275, 355:390]

            lin_img = crop_img.reshape(-1)

            #hist, bins = np.histogram(crop_img.ravel(), 40)

            frames.append(lin_img)

            if is_open:
                labels.append(1)
            else:
                labels.append(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return frames, labels


open_frames = []
open_labels = []
closed_frames = []
closed_labels = []

test_open_frames = []
test_open_labels = []
test_closed_frames = []
test_closed_labels = []

for video_path in training_closed:
    f, l = get_data(video_path, is_open=False)
    closed_frames.extend(f)
    closed_labels.extend(l)

for video_path in training_open:
    f, l = get_data(video_path, is_open=True)
    open_frames.extend(f)
    open_labels.extend(l)

for video_path in test_closed:
    f, l = get_data(video_path, is_open=False)
    test_closed_frames.extend(f)
    test_closed_labels.extend(l)

for video_path in test_open:
    f, l = get_data(video_path, is_open=True)
    test_open_frames.extend(f)
    test_open_labels.extend(l)

'''open_frames = np.array(open_frames)
open_labels = np.array(open_labels)
closed_frames = np.array(closed_frames)
closed_labels = np.array(closed_labels)

test_open_frames = np.array(test_open_frames)
test_open_labels = np.array(test_open_labels)
test_closed_frames = np.array(test_closed_frames)
test_closed_labels = np.array(test_closed_labels)'''

data_x = open_frames + closed_frames
data_y = open_labels + closed_labels
test_x = test_open_frames + test_closed_frames
test_y = test_open_labels + test_closed_labels

data_x = np.array(data_x)
data_y = np.array(data_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#for name, clf in zip(names, classifiers):
#    clf.fit(data_x, data_y)
#    score = clf.score(test_x, test_y)
#    print(name + ":" + str(score))
clf = SVC(kernel="linear", C=0.025)
clf.fit(data_x, data_y)
score = clf.score(test_x, test_y)
print("Score:" + str(score))


cap = cv2.VideoCapture(URL)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        crop_img = frame[250:275, 355:390]
        cv2.imshow("Crop", crop_img)
        cv2.waitKey(1)
        open = clf.predict(np.array([crop_img.reshape(-1)]))
        print(open)


