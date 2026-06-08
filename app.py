from pathlib import Path

import cv2
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_IMAGE = BASE_DIR / "images" / "jikri.jpg"
PROTO_FILE = MODEL_DIR / "pose_deploy.prototxt"
WEIGHTS_FILE = MODEL_DIR / "pose_iter_584000.caffemodel"

N_POINTS = 25

BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24,
    "Background": 25,
}

POSE_PAIRS = [
    ["Neck", "MidHip"],
    ["Neck", "RShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["Neck", "LShoulder"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["MidHip", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["MidHip", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
]


@st.cache_resource
def load_model():
    if not PROTO_FILE.exists():
        raise FileNotFoundError(f"File prototxt tidak ditemukan: {PROTO_FILE}")
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"File caffemodel tidak ditemukan: {WEIGHTS_FILE}")

    return cv2.dnn.readNetFromCaffe(str(PROTO_FILE), str(WEIGHTS_FILE))


def read_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gambar tidak bisa dibaca. Gunakan file JPG, JPEG, atau PNG.")
    return image


def read_default_image():
    image = cv2.imread(str(DEFAULT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Gambar default tidak ditemukan: {DEFAULT_IMAGE}")
    return image


def round_to_stride(value, stride=8):
    return max(stride, int(round(value / stride) * stride))


def predict_pose(net, image, input_width):
    height, width = image.shape[:2]
    input_width = round_to_stride(input_width)
    input_height = round_to_stride((height / width) * input_width)

    blob = cv2.dnn.blobFromImage(
        image,
        1.0 / 255,
        (input_width, input_height),
        (0, 0, 0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    return net.forward()


def create_heatmap(output, image_shape):
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for i in range(N_POINTS):
        pmap = output[0, i, :, :]
        pmap = cv2.resize(pmap, (width, height))
        heatmap = np.maximum(heatmap, pmap)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def extract_keypoints(output, image_shape, threshold):
    height, width = image_shape[:2]
    output_height, output_width = output.shape[2:4]
    points = []

    for i in range(N_POINTS):
        prob_map = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(prob_map)

        x = int((width * point[0]) / output_width)
        y = int((height * point[1]) / output_height)

        if prob > threshold:
            points.append((x, y))
        else:
            points.append(None)

    return points


def draw_keypoints(image, points):
    keypoint_image = image.copy()

    for index, point in enumerate(points):
        if point is None:
            continue

        cv2.circle(keypoint_image, point, 6, (255, 0, 255), thickness=-1)
        cv2.putText(
            keypoint_image,
            str(index),
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return keypoint_image


def draw_pose(image, points):
    pose_image = image.copy()

    for point in points:
        if point is not None:
            cv2.circle(pose_image, point, 5, (255, 0, 255), thickness=-1)

    for pair in POSE_PAIRS:
        part_from = BODY_PARTS[pair[0]]
        part_to = BODY_PARTS[pair[1]]

        if points[part_from] and points[part_to]:
            cv2.line(pose_image, points[part_from], points[part_to], (0, 255, 0), 3)

    return pose_image


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_keypoint_legend():
    body_parts_by_index = {index: name for name, index in BODY_PARTS.items() if name != "Background"}
    return [
        {"Index": index, "Body Part": body_parts_by_index[index]}
        for index in range(N_POINTS)
    ]


def main():
    st.set_page_config(page_title="Human Pose Detection", layout="wide")

    st.title("Human Pose Detection")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], max_upload_size=20)
        threshold = st.slider("Threshold", 0.01, 1.00, 0.10, 0.01)
        input_size = st.select_slider("Input width", options=[256, 320, 368, 432, 512], value=368)
        show_heatmap = st.toggle("Tampilkan heatmap", value=True)
        show_keypoints = st.toggle("Tampilkan keypoints", value=True)

    try:
        net = load_model()
        image = read_uploaded_image(uploaded_file) if uploaded_file else read_default_image()
        output = predict_pose(net, image, input_size)
        points = extract_keypoints(output, image.shape, threshold)

        pose_image = draw_pose(image, points)
        keypoint_image = draw_keypoints(image, points)

        st.subheader("Keypoints dan Pose")
        columns = st.columns(2)

        with columns[0]:
            if show_keypoints:
                st.image(bgr_to_rgb(keypoint_image), caption="Keypoints", use_container_width=True)
            else:
                st.image(bgr_to_rgb(image), caption="Original", use_container_width=True)

        with columns[1]:
            st.image(bgr_to_rgb(pose_image), caption="Pose", use_container_width=True)

        st.subheader("Legend Keypoints")
        legend_columns = st.columns(5)
        legend = get_keypoint_legend()

        for column_index, column in enumerate(legend_columns):
            start = column_index * 5
            end = start + 5
            with column:
                st.dataframe(legend[start:end], hide_index=True, use_container_width=True)

        if show_heatmap:
            st.subheader("Heatmap")
            heatmap = create_heatmap(output, image.shape)
            heatmap_overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
            heatmap_column, _ = st.columns(2)
            with heatmap_column:
                st.image(bgr_to_rgb(heatmap_overlay), caption="Heatmap", use_container_width=True)

    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()