from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\user\Desktop\Hana Studio\temp\pngtree-cartoon-for-kids-without-background-png-image_14227825-removebg-preview_white_layer.png").convert("L")
arr = np.array(img)
unique, counts = np.unique(arr, return_counts=True)

for val, count in zip(unique, counts):
    print(f"값: {val}, 개수: {count}")
