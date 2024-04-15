from PIL import Image, ImageDraw
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from torch import stack, tensor, float16
from generator.singleton import SingletonMeta
from utils.constants import (
    FASHIONOPEDIA_CATEGORIES,
    YOLO_MODEL_NAME,
    YOLO_FEAT_EXTRACTOR,
)


def get_label(i):
    return FASHIONOPEDIA_CATEGORIES[i]


def filter_predictions(outputs, threshold=0.8):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    filtered = probas.max(-1).values > threshold
    classes = [get_label(p.argmax()) for p in probas[filtered]]
    return classes


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * tensor([img_w, img_h, img_w, img_h], dtype=float16)
    return b


class ImageFilter(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.feature_extractor = YolosFeatureExtractor.from_pretrained(YOLO_FEAT_EXTRACTOR)
        self.model = YolosForObjectDetection.from_pretrained(YOLO_MODEL_NAME)
        self.threshold = 0.5


    def filter(self, image: Image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        classes = filter_predictions(outputs, threshold=0.6)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        filtered = probas.max(-1).values > self.threshold

        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, filtered].cpu(), image.size)
        return classes, bboxes_scaled
