import torchvision
from PIL import Image

tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([300, 300], Image.BICUBIC),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
])
img = Image.open('/media/ian/WD/datasets/total_forge/SP/test_and_train/test/images/001221.tif')
img = tf(img)
print(img)
