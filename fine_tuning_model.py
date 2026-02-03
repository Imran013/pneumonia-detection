import torch
import torch.nn as nn
from torchvision import models

class Config:
    """Placeholder"""
    pass

class PneumoniaCNN(nn.Module):
    """Fine-tuned ResNet50 model with FIXED bugs"""
    
    def __init__(self, config, freeze_strategy: str = 'partial'):
        super(PneumoniaCNN, self).__init__()

        self.config = config
        self.freeze_strategy = freeze_strategy

        print(f'🔧 Transfer learning strategy: {freeze_strategy.upper()}')

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        # Apply freezing strategy
        if freeze_strategy == 'all':
            self._freeze_all_layers()
        elif freeze_strategy == 'partial':
            self._freeze_partial_layers()
        elif freeze_strategy == 'none':
            self._freeze_no_layers()
        else:
            raise ValueError(f"Unknown strategy: {freeze_strategy}")
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features

        # Better classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # ➕ Batch norm əlavə edildi
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(512, config.num_classes)
        )

        self._initialize_new_layers()

    def _freeze_all_layers(self):
        """Freeze all backbone layers"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())  # 🔧 FIX: numel() əlavə edildi
        frozen = total - trainable

        print(f"  ✅ Trainable params: {trainable:,}")
        print(f"  ❄️  Frozen params: {frozen:,}")

    def _freeze_partial_layers(self):
        """Freeze early layers (conv1, bn1, layer1, layer2)"""
        layers_to_freeze = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2
        ]

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())  # 🔧 FIX
        frozen = total - trainable

        print(f"  ✅ Trainable params: {trainable:,}")
        print(f"  ❄️  Frozen params: {frozen:,}")

    def _freeze_no_layers(self):
        """Don't freeze any layers"""
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  ✅ Trainable params: {trainable:,}")

    def _initialize_new_layers(self):
        """Initialize new FC layers with Xavier initialization"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # ✨ Xavier initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers: int = 1):
        """Progressive unfreezing of layers"""
        layers = [
            self.backbone.layer4,
            self.backbone.layer3,
            self.backbone.layer2,
            self.backbone.layer1
        ]
        
        for i in range(min(num_layers, len(layers))):
            layer_name = f"layer{4-i}"
            print(f"🔓 Unfreezing {layer_name}")

            for param in layers[i].parameters():
                param.requires_grad = True
        
        # Print updated stats
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  ✅ Trainable params after unfreezing: {trainable:,} / {total:,}")


class FineTuningOptimizer:
    """Optimizer with differential learning rates"""
    
    @staticmethod
    def create_optimizer(model: PneumoniaCNN, config):
        """Create optimizer with different LR for pretrained vs new layers"""
        pretrained_params = []
        new_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)

        # Differential learning rates
        optimizer = torch.optim.AdamW([  # ✨ AdamW istifadə edildi (better than Adam)
            {
                'params': pretrained_params,
                'lr': config.learning_rate * 0.1,  # 10x kiçik LR
                'name': 'pretrained_layers'
            },
            {
                'params': new_params,
                'lr': config.learning_rate,
                'name': 'new_fc_layers'
            }
        ], weight_decay=config.weight_decay)

        print(f"\n📊 Optimizer configuration:")
        print(f"  Pretrained layers LR: {config.learning_rate * 0.1:.2e}")
        print(f"  New FC layers LR: {config.learning_rate:.2e}")
        print(f"  Weight decay: {config.weight_decay:.2e}")

        return optimizer


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}