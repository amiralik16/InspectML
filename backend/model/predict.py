import torch
from .helpers import *
import warnings
from PIL import Image
from torchvision import transforms
from pathlib import Path
#from torchsummary import summary

def image_transform(image):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    imagetensor = test_transforms(image)
    return imagetensor


def predict_image(image, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = Path(__file__).parent / "catvdog.pth"
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = load_model(model_path)
    model.eval()
    #summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(image)
    image1 = image[None,:,:,:]
    ps=torch.exp(model(image1))
    topconf, topclass = ps.topk(1, dim=1)
    if topclass.item() == 1:
        return {'class': 'dog', 'confidence': topconf.item()}
    else:
        return {'class': 'cat', 'confidence': topconf.item()}