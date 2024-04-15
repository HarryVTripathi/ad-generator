from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
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


class Filter(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.feature_extractor = YolosFeatureExtractor.from_pretrained(YOLO_FEAT_EXTRACTOR)
        self.model = YolosForObjectDetection.from_pretrained(YOLO_MODEL_NAME)


    def filter(self, image: Image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        classes = filter_predictions(outputs, threshold=0.8)

        return classes