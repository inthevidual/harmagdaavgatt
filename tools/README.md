# Building assets/magda3d.bin

Offline pipeline that fits a 3D face model to `Magda.png` and exports the morph targets used by `magda-face-3d.js`.

Requirements (not committed): a Python venv with `mediapipe numpy scipy pillow`,
ICT-FaceKit's `FaceXModel/` folder (github.com/USC-ICT/ICT-FaceKit, MIT) and
MediaPipe's `face_landmarker.task` model.

```sh
python tools/load_ict.py  <ICT>/FaceXModel work/ict.npz           # parse meshes
python tools/landmarks.py Magda.png face_landmarker.task work/lm.npz  # 478 landmarks + her expression
python tools/fit.py       work                                      # identity + pose fit
python tools/export.py    work assets/magda3d.bin                   # morph targets → binary
```

Expression recipes (smile / frown) are the `SMILE` and `FROWN` dicts in `export.py`;
the values are ICT/ARKit blendshape weights. Re-run `export.py` after changing them.
Note: `fit.py` contains the expression MediaPipe measured on this photo (mouthPress etc.);
re-run `landmarks.py` and update that table if the photo changes.

# Building assets/magda.avif / magda.webp

The cut-out still carries the old red backdrop in its edge pixels. `defringe.py`
repaints them so the portrait sits cleanly on the black page:

```sh
python tools/defringe.py Magda.png work/magda-clean.png
magick work/magda-clean.png -resize 1200x work/magda-1200.png
avifenc -q 60 --speed 4 work/magda-1200.png assets/magda.avif
cwebp -q 80 -alpha_q 90 work/magda-1200.png -o assets/magda.webp
```
