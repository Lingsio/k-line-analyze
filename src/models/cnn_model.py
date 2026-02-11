
import torch
import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, input_channels=3, arch='resnet18'):
        super(CNNModel, self).__init__()
        self.arch = arch
        self.input_channels = input_channels
        self.embedding = None # Hook to store embedding
        
        if arch == 'simple_cnn':
            # LeNet-style simple model
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # 32x32
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # 16x16
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # 8x8
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)  # 4x4
            )
            # Embedding dim = 128 * 4 * 4 = 2048
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.embedding_layer_name = 'fc1'
            
        elif arch == 'resnet18':
            # Use ResNet18 as backbone
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # Modify first layer - adapt to input size automatically
            # For 256x256, we can use stride=2, for 128x128 stride=1 is fine
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(512, num_classes)
            
        elif arch == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            # EfficientNet first layer is typically stride 2.
            # features[0][0] is Conv2dNormActivation -> 0 is Conv2d
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(input_channels, original_conv.out_channels, 
                                                     kernel_size=3, stride=1, padding=1, bias=False)
            # Classifier is sequences of dropout + linear
            self.backbone.classifier[1] = nn.Linear(1280, num_classes)
            
        elif arch == 'vit_b_16':
            # Vision Transformer
            # Requires 224x224 input usually.
            self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            
            # Handle input channels if not 3
            if input_channels != 3:
                # Conv projection is self.backbone.conv_proj
                original_proj = self.backbone.conv_proj
                self.backbone.conv_proj = nn.Conv2d(input_channels, original_proj.out_channels, 
                                                    kernel_size=original_proj.kernel_size, 
                                                    stride=original_proj.stride, bias=False)
            
            # Head
            self.backbone.heads.head = nn.Linear(768, num_classes)
            
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        if self.arch == 'simple_cnn':
            x = self.features(x)
            x = x.view(x.size(0), -1) # Flatten
            # Store embedding (output of first FC, before logits)
            # To do this cleanly, we split classifier
            emb = self.classifier[0](x)
            emb = self.classifier[1](emb)
            self.embedding = emb # Hook for retrieval
            emb = self.classifier[2](emb)
            logits = self.classifier[3](emb)
            return logits
            
        elif self.arch == 'resnet18':
            # ResNet forward logic to capture embedding
            # x = self.backbone(x) # This runs everything
            # We need to hook before fc
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            self.embedding = x # 512 dim
            x = self.backbone.fc(x)
            return x
            
        elif self.arch == 'efficientnet_b0':
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            self.embedding = x # 1280 dim
            x = self.backbone.classifier(x)
            return x
            
        elif self.arch == 'vit_b_16':
            x = self.resize(x)
            x = self.backbone._process_input(x)
            
            # Expand standard forward to capture cls token
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)
            x = x[:, 0] # Class token
            self.embedding = x # 768 dim
            x = self.backbone.heads(x)
            return x


    def extract_features(self, x):
        """
        Run forward pass but stop before classification head.
        Returns the embedding vector.
        """
        with torch.no_grad():
            if self.arch == 'simple_cnn':
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier[0](x)
                x = self.classifier[1](x)
                return x
                
            elif self.arch == 'resnet18':
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)

                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)

                x = self.backbone.avgpool(x)
                x = torch.flatten(x, 1)
                return x
                
            elif self.arch == 'efficientnet_b0':
                x = self.backbone.features(x)
                x = self.backbone.avgpool(x)
                x = torch.flatten(x, 1)
                return x
                
            elif self.arch == 'vit_b_16':
                x = self.resize(x)
                x = self.backbone._process_input(x)
                n = x.shape[0]
                batch_class_token = self.backbone.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.backbone.encoder(x)
                x = x[:, 0]
                return x
