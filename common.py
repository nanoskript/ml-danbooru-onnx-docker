from PIL import Image
from torchvision import transforms


def prepare_image(image: Image, image_size: int):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return transform(image.convert("RGB")).unsqueeze(0)
