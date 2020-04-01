from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class ConcreteDamages(CocoDataset):

    CLASSES = ('crack', 'effl', 'rebar', 'spll', )
