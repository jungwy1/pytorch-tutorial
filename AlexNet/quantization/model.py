import torch.nn as nn

# ============================================================================
# AlexNet architecture vs. the paper
# ----------------------------------------------------------------------------
# SAME as paper:
#   - 5 conv + 3 FC layers; ReLU after every conv and the first two FC layers.
#   - Overlapping max pooling (kernel 3, stride 2) after conv1, conv2, conv5
#     (Section 3.4; kernel > stride so windows overlap).
#   - Grouped convs (groups=2) on conv2/conv4/conv5, conv3 is fully connected.
#     This reproduces the original 2-GPU split (= grouped convolution).
#   - Weight init N(0, 0.01); bias=1 on conv2/4/5 and FC1/FC2, bias=0 elsewhere
#     (Section 5; bias=1 gives the ReLUs positive inputs early on).
#   - Dropout(0.5) on the first two FC layers (Section 4.2).
#
# DIFFERENT from paper:
#   - Input is 227, not 224. 11x11/stride-4 conv1 only yields a clean 55x55
#     from 227 (well-known 224-vs-227 issue).
#   - LRN (local response normalization, Section 3.3) is OMITTED. ReLU is
#     non-saturating so it is not needed to prevent saturation; costs ~1% top-1.
#   - Output layer is num_classes=10 (Imagenette), not 1000 (ImageNet).
#   - No softmax in the model: raw logits go to nn.CrossEntropyLoss, which
#     applies log-softmax internally (avoids double softmax, numerically stable).
# ============================================================================

class AlexNet(nn.Module):
    # Weight init from the paper (Section 5)
    def _init_weights(self):
        # all conv/linear weights ~ N(0, 0.01), all biases 0
        for m in self.model:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
        # bias = 1 for conv2, conv4, conv5 and FC1, FC2 (feeds ReLUs positive
        # inputs early). Collect by type so exact Sequential indices don't matter.
        convs = [m for m in self.model if isinstance(m, nn.Conv2d)]
        linears = [m for m in self.model if isinstance(m, nn.Linear)]
        for m in (convs[1], convs[3], convs[4], linears[0], linears[1]):
            nn.init.constant_(m.bias, 1.0)

    def __init__(self, num_classes = 10, dropout = 0.5):
        super().__init__()
        self.model = nn.Sequential(
            # CONV-1
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(), # 227 -> 55
            # MaxPool-1
            nn.MaxPool2d(kernel_size=3, stride=2), # 55 -> 27
            # CONV-2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), nn.ReLU(), # 27 -> 27
            # MaxPool-2
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13
            # CONV-3
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), # 13 -> 13
            # CONV-4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2), nn.ReLU(), # 13 -> 13
            # CONV-5
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2), nn.ReLU(), # 13 -> 13
            # MaxPool-3
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
            # Flatten
            nn.Flatten(start_dim=1),
            # FC-1
            nn.Linear(256*6*6, 4096), nn.ReLU(), nn.Dropout(p=dropout),       
            # FC-2
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=dropout),  
            # FC-3
            nn.Linear(4096, num_classes)
        )
        self._init_weights()    # apply the paper's weight/bias init

    def forward(self, x):
        logits = self.model(x)
        return logits
    


# validation for the above code
if __name__ == "__main__":
    import torch
    model = AlexNet()
    # validate the weigths initialization of the model
    convs = [m for m in model.model if isinstance(m, nn.Conv2d)]
    print(convs[0].weight.std().item())     # ~0.01 (conv1)
    print(convs[0].bias.unique())           # tensor([0.]) (conv1)
    print(convs[1].bias.unique())           # tensor([1.]) (conv2)
    # print a summary of the model
    x = torch.randn(2, 3, 227, 227)
    print(model(x).shape)                   # torch.Size([2, 10])
    from torchinfo import summary
    summary(model, input_size=(1, 3, 227, 227))
