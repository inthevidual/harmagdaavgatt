import sys, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
img_path, model, out = sys.argv[1:4]
opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model),
    output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
det = vision.FaceLandmarker.create_from_options(opts)
im = mp.Image.create_from_file(img_path)
r = det.detect(im)
lm = np.array([[p.x*im.width, p.y*im.height, p.z*im.width] for p in r.face_landmarks[0]], np.float32)
bs = {c.category_name: round(c.score,3) for c in r.face_blendshapes[0] if c.score > 0.05}
np.savez(out, lm=lm, M=np.array(r.facial_transformation_matrixes[0]), W=im.width, H=im.height)
print(lm.shape, im.width, im.height); print('blendshapes>0.05:', bs)
