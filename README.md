# Human Pose Detection

This project implements human pose estimation using the OpenPose deep learning model with OpenCV DNN. The application is built with Streamlit so pose prediction can be performed through a simple web interface.

This project is based on material from *Python Image Processing Cookbook* by Sandipan Dey and the LearnOpenCV reference on deep learning based human pose estimation.

## Live Demo

The deployed application is available at:

https://humanposedetect.streamlit.app/

## Features

- Upload images through Streamlit.
- Predict human body keypoints using the OpenPose BODY_25 model.
- Visualize keypoints with index numbers.
- Visualize the pose skeleton.
- Visualize the confidence map heatmap.
- Display a BODY_25 keypoint legend.
- Adjust threshold and input width from the sidebar.

## Model Used

This project uses the OpenPose BODY_25 model:

```text
models/pose_deploy.prototxt
models/pose_iter_584000.caffemodel
```

The BODY_25 model produces 25 body keypoints and 1 background channel. In this application, all keypoints are still predicted and displayed, but the skeleton lines only connect the main body parts. Eye, ear, toe, and heel keypoints are still shown, but they are not connected by skeleton lines.


## How to Use

1. Open the deployed Streamlit application.
2. Upload an image in JPG, JPEG, or PNG format.
3. Adjust the `Threshold` value if there are too few or too many keypoints.
4. Adjust `Input width` to change the model input size.
5. View the result in the `Keypoints and Pose` section.
6. Use the `Legend Keypoints` section to understand what each index number represents.
7. View the `Heatmap` section to inspect the confidence map visualization.

## BODY_25 Keypoint List

| Index | Body Part |
| --- | --- |
| 0 | Nose |
| 1 | Neck |
| 2 | RShoulder |
| 3 | RElbow |
| 4 | RWrist |
| 5 | LShoulder |
| 6 | LElbow |
| 7 | LWrist |
| 8 | MidHip |
| 9 | RHip |
| 10 | RKnee |
| 11 | RAnkle |
| 12 | LHip |
| 13 | LKnee |
| 14 | LAnkle |
| 15 | REye |
| 16 | LEye |
| 17 | REar |
| 18 | LEar |
| 19 | LBigToe |
| 20 | LSmallToe |
| 21 | LHeel |
| 22 | RBigToe |
| 23 | RSmallToe |
| 24 | RHeel |

## Prediction Notes

Prediction quality may decrease if:

- the person is too small in the image,
- the image contains more than one person,
- some body parts are cropped or occluded,
- the pose is too extreme,
- the lighting is unclear,
- the threshold value is too high or too low.

For more stable results, use an image with one person, a clearly visible body, and a clear pose.

## References

- Sandipan Dey, *Python Image Processing Cookbook*.
- LearnOpenCV: Deep Learning based Human Pose Estimation using OpenCV.
- OpenPose BODY_25 model by CMU Perceptual Computing Lab.