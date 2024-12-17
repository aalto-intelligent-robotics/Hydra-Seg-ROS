# from ultralytics.utils.plotting import colors
# import numpy as np
#
# c = []
# for i in range(80):
#     color = colors(i, True)
#     while color in c:
#         color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
#     c.append(color)
#     print(f"{i}: ({','.join(str(x) for x in list(color))}),")
from hydra_seg_ros.utils.labels import COCO_COLORS, COCO_FULL_CATEGORIES

for k, v in COCO_COLORS.items():
    print(f"{COCO_FULL_CATEGORIES[k]},{','.join(str(x) for x in v)},255,{k}")
