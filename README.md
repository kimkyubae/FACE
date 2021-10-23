# Face Tracking (얼굴 추적) - 동영상에서 얼굴을 탐지 한뒤 추적
# Face Recognatiom (얼굴 인식) - 사진에 얼굴을 인식한 뒤 찾는것
# Face Detecting (얼굴 탐지) - 사진 한 장에서 얼굴의 위치를 찾음
import dlib, cv2 
# dlib - Face Detecting, Face Recognatiom
# cv2 - opencv (이미지 작업)

import numpy as np
# numpy - 행렬 연산 작업

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# Matplotlib - 결과물 출력하기 위해 쓴다 

detector = dlib.get_frontal_face_detector()
# dlib.get_frontal_face_detecto - 얼굴 탐지 모델

sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
# dlib.shape_predictor - 얼굴 랜드마크 탐지 모델

facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
# face_recognition_model_v1 - 얼굴 인식 모델

# 얼굴을 찾는 함수
def find_faces(img): # input을 (RGB)img로 받는다 
    dets = detector(img, 1)

    if len(dets) == 0: # 얼굴을 찾지 못한다 
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], [] 
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    # 얼굴에 68개의 점을 지정
    
    for k, d in enumerate(dets): # 얼굴을 찾을때마다 루프를 돈다.
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        # 얼굴의 왼쪽, 위쪽, 오른쪽, 아래쪽 좌표를 넣어준다.

        rects.append(rect)
        # rects 변수에 쌓아준다.
 
        shape = sp(img, d)
        # img, d = 사각형을 넣으면 shape에 68개의 점을 넣는다.

        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape) # 랜드마크 결과물을 shapes에 쌓는다.
        
    return rects, shapes, shapes_np


def encode_faces(img, shapes): # face를 인코드하는 함수
    #face 랜드마크 정보를 encoder에 넣어주면 128개 vector가 나옴
    #128개의 특징들로 사람을 구별한다.
   
    face_descriptors = [] # 결과값 저장
    for shape in shapes: # 랜드마크의 집합 크기 만큼 루프를 돈다.
        face_descriptor = facerec.compute_face_descriptor(img, shape) # face_recognition_model_v1 돌려준다.
        face_descriptors.append(np.array(face_descriptor))
        # 데이터를 넘파이 값으로 받아서 descriptor에 쌓아줌
    return np.array(face_descriptors)
# 1. face detection (얼굴찾기)
# 2. face Landmark Detecting (얼굴 랜드마크 찾기) - shapes 저장
# 3. face encoding (얼굴 인코딩) - 각 사람의 랜드마크와 전체 이미지를 넣는다


# 얼굴을 찾는 함수
def find_faces(img): # input을 (RGB)img로 받는다 
    dets = detector(img, 1)

    if len(dets) == 0: # 얼굴을 찾지 못한다 
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], [] 
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    # 얼굴에 68개의 점을 지정
    
    for k, d in enumerate(dets): # 얼굴을 찾을때마다 루프를 돈다.
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        # 얼굴의 왼쪽, 위쪽, 오른쪽, 아래쪽 좌표를 넣어준다.

        rects.append(rect)
        # rects 변수에 쌓아준다.
 
        shape = sp(img, d)
        # img, d = 사각형을 넣으면 shape에 68개의 점을 넣는다.

        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape) # 랜드마크 결과물을 shapes에 쌓는다.
        
    return rects, shapes, shapes_np


def encode_faces(img, shapes): # face를 인코드하는 함수
    #face 랜드마크 정보를 encoder에 넣어주면 128개 vector가 나옴
    #128개의 특징들로 사람을 구별한다.
   
    face_descriptors = [] # 결과값 저장
    for shape in shapes: # 랜드마크의 집합 크기 만큼 루프를 돈다.
        face_descriptor = facerec.compute_face_descriptor(img, shape) # face_recognition_model_v1 돌려준다.
        face_descriptors.append(np.array(face_descriptor))
        # 데이터를 넘파이 값으로 받아서 descriptor에 쌓아줌
    return np.array(face_descriptors)
# 1. face detection (얼굴찾기)
# 2. face Landmark Detecting (얼굴 랜드마크 찾기) - shapes 저장
# 3. face encoding (얼굴 인코딩) - 각 사람의 랜드마크와 전체 이미지를 넣는다

img_paths = {
    'kv': 'img/kv.jpg', # 케빈
    'mom': 'img/mom.jpg', # 엄마
    'tf1': 'img/tf1.jpg', # 도둑 1
    'tf2': 'img/tf2.jpg' # 도둑2
}

descs = {
    'kv': None,
    'mom': None,
    'tf1': None,
    'tf2': None
}

for name, img_path in img_paths.items(): 
    # img_path 안의 이미지를 cv2.imread()로 읽어준다
    # bgr로 나와서 rgb로 바꿔줘야한다. cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb) # 얼굴을 찾은 뒤 shape => landmark를 받아온뒤
    descs[name] = encode_faces(img_rgb, img_shapes)[0]
    # encode_faces 함수에 전체 이미지 각 사람의 land mark를 넣어줌

np.save('img/descs.npy', descs) 
print(descs) # 각 사람의 numpy 행렬을 출력

img_bgr = cv2.imread('img/homealone.jpg') 
# 이미지 파일을 imread()읽는다.
# 얼굴을 찾은뒤 인코드 하고 128개의 벡터를 읽고 비교해서 누구인지 판단

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# bgr => rgb로 변경

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)
# 변경한 결과를 descriptors로 받아온다.

ig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors): # descriptors 수 만큼 루프를 돈다.
    
    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1)
       # np.linalg.norm 벡터 사이의 거리를 구한다. (얼굴 유사도 판단)

        if dist < 0.6: # dist(거리)가 0.6 보다 작으면 찾았다는 의미
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break 
    
    if not found: # 0.6보다 크면  unknown이라 나온다.
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('result/output.png')
plt.show()
