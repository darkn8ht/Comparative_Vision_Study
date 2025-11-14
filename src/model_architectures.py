import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

def _freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    
def get_classification_model(model_name, num_classes):
    
    if model_name == 'DenseNet-121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        _freeze_layers(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        _freeze_layers(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'ResNet-50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        _freeze_layers(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'MobileNet-V2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        _freeze_layers(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'VGG-16': 
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        _freeze_layers(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    for param in list(model.parameters())[-50:]: 
        param.requires_grad = True
            
    return model

def get_model(task_type, model_name, num_classes):
    
    if task_type == 'segmentation':
        encoder_map = {
            'DenseNet-121': 'densenet121',
            'EfficientNet-B0': 'efficientnet-b0',
            'ResNet-50': 'resnet50',
            'MobileNet-V2': 'mobilenet_v2',
            'VGG-16': 'vgg16'
        }
        encoder_name = encoder_map.get(model_name)
        if encoder_name is None:
            raise ValueError(f"Encoder {model_name} not supported for segmentation.")
            
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None
        )

    elif task_type == 'classification':
        return get_classification_model(model_name, num_classes)
    else:
        raise ValueError(f"Invalid model/task combination: {model_name}/{task_type}")