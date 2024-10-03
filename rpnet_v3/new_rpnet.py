import torch
import torch.nn as nn
import torchvision.ops as ops


prov_Num, alpha_Num, ad_Num = 38, 25, 35


class SqueezeAndExcitation(nn.Module): 
    def __init__(self, in_channel, ratio=4): 
        super(SqueezeAndExcitation, self).__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.network = nn.Sequential(
            nn.Linear(in_channel, in_channel//ratio),
            nn.ReLU(), 
            nn.Linear(in_channel//ratio, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x): 
        b, c, _, _ = x.shape 
        p = self.avg_pool(x) 
        p = p.view(b, c) 
        p = self.network(p) 
        p = p.view(b, c, 1, 1) 
        x = x * p 
        return x 
    

class BasicBlock(nn.Module):  
    def __init__(self, in_channels, out_channels):
      super(BasicBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
      self.bn1 = nn.BatchNorm2d(out_channels)
      self.relu = nn.ReLU()
      self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
      self.bn2 = nn.BatchNorm2d(out_channels)
      self.downsample =  nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2),
          nn.BatchNorm2d(out_channels),
      )
  
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x + identity)
    

class Detection_Module(nn.Module):
    def __init__(self, num_Points = 4):
        super(Detection_Module, self).__init__()
        hidden0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        hidden1 = BasicBlock(in_channels=64, out_channels=64)
        hidden2 = BasicBlock(in_channels=64, out_channels=128)
        hidden3 = BasicBlock(in_channels=128, out_channels=256)
        hidden4 = BasicBlock(in_channels=256, out_channels=512)
        hidden5 = BasicBlock(in_channels=512, out_channels=512)

        hidden6 = nn.Sequential(
            SqueezeAndExcitation(in_channel=512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.features = nn.Sequential(
            hidden0,
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_Points),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = torch.flatten(x1, start_dim=1)
        x = self.classifier(x11)
        return x
    

class Recognition_Module(nn.Module):
    def __init__(self, device = 'cpu', path = None):
        super(Recognition_Module, self).__init__()
        self.device = device
        self.detection_module = Detection_Module()
        self.load_detection_module(path)

        self.preprocess = nn.Sequential(
            SqueezeAndExcitation(in_channel=512),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, prov_Num),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, alpha_Num),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )

    def load_detection_module(self, path):
        self.detection_module.load_state_dict(torch.load(path, map_location=self.device))

    def forward(self, x):
        x0 = self.detection_module.features[0](x)
        x1 = self.detection_module.features[1](x0)
        x2 = self.detection_module.features[2](x1)
        x3 = self.detection_module.features[3](x2)
        x4 = self.detection_module.features[4](x3)
        x5 = self.detection_module.features[5](x4)
        x6 = self.detection_module.features[6](x5)
        x6 = torch.flatten(x6, start_dim=1)
        boxLoc = self.detection_module.classifier(x6) # B x 4
 
        h, w = x.shape[2], x.shape[3]
        batch_size = boxLoc.shape[0]
        postfix = torch.tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]], requires_grad=False, dtype=torch.float).to(self.device) # 4 x 4
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1) # B x 4
        p = torch.tensor([[w,0,0,0],[0,h,0,0],[0,0,w,0],[0,0,0,h]], requires_grad=False, dtype=torch.float).to(self.device) # 4 x 4
        indices = torch.arange(batch_size).unsqueeze(1).to(self.device) # B x 1
        roi = torch.cat((indices, boxNew.mm(p)), dim = 1) # B x 5

        pool0 = ops.roi_pool(x0, roi, (16, 8), 1/2)
        pool1 = ops.roi_pool(x1, roi, (16, 8), 1/4)
        pool2 = ops.roi_pool(x2, roi, (16, 8), 1/8)
        pool3 = ops.roi_pool(x3, roi, (16, 8), 1/16)
        
        pooled_features = torch.cat((pool0, pool1, pool2, pool3), dim = 1) # B x ch x 16 x 8
        pooled_features = self.preprocess(pooled_features)
        pooled_features = torch.flatten(pooled_features, start_dim=1)

        y0 = self.classifier1(pooled_features)
        y1 = self.classifier2(pooled_features)
        y2 = self.classifier3(pooled_features)
        y3 = self.classifier4(pooled_features)
        y4 = self.classifier5(pooled_features)
        y5 = self.classifier6(pooled_features)
        y6 = self.classifier7(pooled_features)

        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]