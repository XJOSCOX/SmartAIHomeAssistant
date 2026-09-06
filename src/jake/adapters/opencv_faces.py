"""Local YuNet and SFace adapters; all model-specific image translation lives here."""

from hashlib import file_digest
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.domain import BoundingBox, Frame
from jake.identity import IdentityError, face_normalize
from jake.identity_config import IdentityConfig
from jake.identity_domain import FaceDetection, FaceEmbedding, FaceQuality

MODEL_ID = "opencv-sface-2021dec-128"


def crop(frame: Frame, box: BoundingBox) -> tuple[NDArray[np.uint8], int, int]:
    left, top = floor(box.left * frame.width), floor(box.top * frame.height)
    right, bottom = ceil(box.right * frame.width), ceil(box.bottom * frame.height)
    rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    return np.ascontiguousarray(rgb[top:bottom, left:right, ::-1]), left, top


class YuNetFaceDetector:
    def __init__(self, config: IdentityConfig) -> None:
        if not Path(config.detector_model).is_file():
            raise IdentityError("missing local YuNet model; no automatic download")
        try:
            self._model = cv2.FaceDetectorYN.create(
                config.detector_model,
                "",
                (320, 320),
                0.6,
                0.3,
                5000,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except cv2.error as exc:
            raise IdentityError("cannot load local YuNet model") from exc

    def detect(self, frame: Frame, region: BoundingBox) -> tuple[FaceDetection, ...]:
        bgr, x, y = crop(frame, region)
        try:
            self._model.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, rows = self._model.detect(bgr)
            if rows is None:
                return ()
            faces = []
            for row in rows:
                if len(row) != 15 or not np.all(np.isfinite(row)):
                    continue
                face_left, t, w, h = (float(v) for v in row[:4])
                clipped = (
                    face_left <= 0
                    or t <= 0
                    or face_left + w >= bgr.shape[1]
                    or t + h >= bgr.shape[0]
                )
                left, top = max(0.0, face_left + x) / frame.width, max(0.0, t + y) / frame.height
                right, bottom = (
                    min(frame.width, face_left + x + w) / frame.width,
                    min(frame.height, t + y + h) / frame.height,
                )
                try:
                    points = tuple(
                        ((float(row[i]) + x) / frame.width, (float(row[i + 1]) + y) / frame.height)
                        for i in range(4, 14, 2)
                    )
                    faces.append(
                        FaceDetection(
                            BoundingBox(left, top, right, bottom), float(row[14]), points, clipped
                        )
                    )
                except ValueError:
                    continue
            return tuple(faces)
        except cv2.error as exc:
            raise IdentityError("local face detection failed") from exc


class SFaceEncoder:
    def __init__(self, config: IdentityConfig) -> None:
        if not Path(config.encoder_model).is_file():
            raise IdentityError("missing local SFace model; no automatic download")
        with Path(config.encoder_model).open("rb") as stream:
            self._model_id = MODEL_ID + ":" + file_digest(stream, "sha256").hexdigest()
        try:
            self._model = cv2.FaceRecognizerSF.create(
                config.encoder_model, "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
            )
        except cv2.error as exc:
            raise IdentityError("cannot load local SFace model") from exc

    def encode(self, frame: Frame, face: FaceDetection) -> FaceEmbedding:
        if len(face.landmarks) != 5:
            raise IdentityError("SFace requires five landmarks")
        bgr, x, y = crop(frame, face.box)
        row = np.array(
            [
                0,
                0,
                bgr.shape[1],
                bgr.shape[0],
                *[
                    v
                    for point in face.landmarks
                    for v in (point[0] * frame.width - x, point[1] * frame.height - y)
                ],
                face.confidence,
            ],
            dtype=np.float32,
        )
        try:
            aligned = self._model.alignCrop(bgr, row)
            values = np.asarray(self._model.feature(aligned), dtype=np.float32).reshape(-1)
            if values.size != 128:
                raise IdentityError("SFace output dimension must be 128")
            return face_normalize(self._model_id, tuple(float(v) for v in values))
        except (cv2.error, ValueError) as exc:
            raise IdentityError("local face encoding failed") from exc


def face_quality(frame: Frame, face: FaceDetection, config: IdentityConfig) -> FaceQuality:
    if face.confidence < config.min_detector_confidence:
        return FaceQuality(False, "low detector confidence")
    if (
        face.clipped
        or min(face.box.left, face.box.top, 1 - face.box.right, 1 - face.box.bottom) <= 0.005
    ):
        return FaceQuality(False, "clipped face")
    bgr, _, _ = crop(frame, face.box)
    if min(bgr.shape[:2]) < config.min_face_pixels:
        return FaceQuality(False, "face too small")
    sharpness = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharpness < config.min_sharpness:
        return FaceQuality(False, "blurred face")
    if len(face.landmarks) != 5:
        return FaceQuality(False, "landmarks unavailable")
    points = np.array([(x * frame.width, y * frame.height) for x, y in face.landmarks])
    eye_mid = (points[0] + points[1]) / 2
    eye_span = float(np.linalg.norm(points[1] - points[0]))
    mouth_mid = (points[3] + points[4]) / 2
    if eye_span < 1 or mouth_mid[1] <= eye_mid[1]:
        return FaceQuality(False, "invalid landmark geometry")
    yaw = float((points[2, 0] - eye_mid[0]) / eye_span)
    pitch = float((points[2, 1] - eye_mid[1]) / (mouth_mid[1] - eye_mid[1]))
    roll = abs(float(points[1, 1] - points[0, 1])) / eye_span
    if abs(yaw) > 0.35 or not 0.2 <= pitch <= 0.85 or roll > 0.3:
        return FaceQuality(False, "extreme pose")
    return FaceQuality(
        True, "accepted", "left" if yaw < -0.08 else "right" if yaw > 0.08 else "center"
    )
