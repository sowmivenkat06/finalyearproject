import os
import torch
import torch.nn as nn

from torchvision.models import *

from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

###########################################################
# Load All Trained Models
###########################################################

def load_all_models():

    models = {}

    #######################################################
    # ResNet50
    #######################################################

    resnet50_model = resnet50(weights=None)

    resnet50_model.fc = nn.Linear(
        resnet50_model.fc.in_features,
        NUM_CLASSES
    )

    resnet50_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "ResNet50.pth"),
            map_location=DEVICE
        )
    )

    resnet50_model.fc = nn.Identity()

    resnet50_model.eval()

    models["ResNet50"] = resnet50_model.to(DEVICE)

    #######################################################
    # ResNet101
    #######################################################

    resnet101_model = resnet101(weights=None)

    resnet101_model.fc = nn.Linear(
        resnet101_model.fc.in_features,
        NUM_CLASSES
    )

    resnet101_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "ResNet101.pth"),
            map_location=DEVICE
        )
    )

    resnet101_model.fc = nn.Identity()

    resnet101_model.eval()

    models["ResNet101"] = resnet101_model.to(DEVICE)

    #######################################################
    # ResNet152
    #######################################################

    resnet152_model = resnet152(weights=None)

    resnet152_model.fc = nn.Linear(
        resnet152_model.fc.in_features,
        NUM_CLASSES
    )

    resnet152_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "ResNet152.pth"),
            map_location=DEVICE
        )
    )

    resnet152_model.fc = nn.Identity()

    resnet152_model.eval()

    models["ResNet152"] = resnet152_model.to(DEVICE)

    #######################################################
    # AlexNet
    #######################################################

    alex_model = alexnet(weights=None)

    alex_model.classifier[6] = nn.Linear(
        alex_model.classifier[6].in_features,
        NUM_CLASSES
    )

    alex_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "AlexNet.pth"),
            map_location=DEVICE
        )
    )

    alex_model.classifier[6] = nn.Identity()

    alex_model.eval()

    models["AlexNet"] = alex_model.to(DEVICE)

    #######################################################
    # VGG16
    #######################################################

    vgg_model = vgg16(weights=None)

    vgg_model.classifier[6] = nn.Linear(
        vgg_model.classifier[6].in_features,
        NUM_CLASSES
    )

    vgg_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "VGG16.pth"),
            map_location=DEVICE
        )
    )

    vgg_model.classifier[6] = nn.Identity()

    vgg_model.eval()

    models["VGG16"] = vgg_model.to(DEVICE)

    #######################################################
    # GoogleNet
    #######################################################

    goog_model = googlenet(
        weights=None,
        aux_logits=False
    )

    goog_model.fc = nn.Linear(
        goog_model.fc.in_features,
        NUM_CLASSES
    )

    goog_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "GoogleNet.pth"),
            map_location=DEVICE
        )
    )

    goog_model.fc = nn.Identity()

    goog_model.eval()

    models["GoogleNet"] = goog_model.to(DEVICE)

    #######################################################
    # InceptionV3
    #######################################################

    inception_model = inception_v3(
        weights=None,
        aux_logits=True
    )

    inception_model.fc = nn.Linear(
        inception_model.fc.in_features,
        NUM_CLASSES
    )

    inception_model.AuxLogits.fc = nn.Linear(
        inception_model.AuxLogits.fc.in_features,
        NUM_CLASSES
    )

    inception_model.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "models", "InceptionV3.pth"),
            map_location=DEVICE
        )
    )

    inception_model.fc = nn.Identity()

    inception_model.eval()

    models["InceptionV3"] = inception_model.to(DEVICE)

    return models
