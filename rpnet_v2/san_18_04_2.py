import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

prov_Num, alpha_Num, ad_Num = 38, 25, 35

class Aggregate_Block(nn.Module):
    def __init__(self, pool):
        super(Aggregate_Block, self).__init__()
        self.pool = pool
        if self.pool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, prev_agg, x):
        x = torch.cat([prev_agg, x], dim=1)
        if self.pool:
            x = self.maxpool(x)
        return x
    

class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)

        self.layer_0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer_1 = model.layer1
        self.layer_2 = model.layer2
        self.layer_3 = model.layer3
        self.layer_4 = model.layer4
        
        del model

        self.agg_block1 = Aggregate_Block(pool=True)
        self.agg_block2 = Aggregate_Block(pool=True)
        self.agg_block3 = Aggregate_Block(pool=True)
        self.agg_block4 = Aggregate_Block(pool=False)

        self.embedd = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Tanh(),
        )

    def forward(self, image):
        # B x 3 X 480 x 480

        x0 = self.layer_0(image)
        # B x 64 X 120 x 120

        x1 = self.layer_1(x0)
        # B x 64 x 120 x 120
        agg1 = self.agg_block1(x0, x1)
        # B x 128 x 60 x 60

        x2 = self.layer_2(x1)
        # B x 128 x 60 x 60
        agg2 = self.agg_block2(agg1, x2)
        # B x 256 x 30 x 30

        x3 = self.layer_3(x2)
        # B x 256 x 30 x 30
        agg3 = self.agg_block3(agg2, x3)
        # B x 512 x 15 x 15

        x4 = self.layer_4(x3)
        # B x 512 x 15 x 15
        agg4 = self.agg_block4(agg3, x4)
        # B x 1024 x 15 x 15

        features = torch.flatten(agg4, start_dim=2)
        features = features.permute(0, 2, 1)
        # B x 225 x 1024

        embeddings = self.embedd(features)
        return embeddings
    
class Attention(nn.Module):
    def __init__(self, dim_i=1024, dim_q=1024, dim_u=None):
        super(Attention, self).__init__()
        self.dim_u = dim_u

        self.image_features = nn.Linear(dim_i, dim_q)
        if self.dim_u:
            self.querry_features = nn.Linear(dim_u, dim_q)
        self.attn_scores = nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim_q, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, V_i, u):
        t = self.image_features(V_i)
        if self.dim_u:
            t1 = self.querry_features(u)
            t = t + t1.unsqueeze(1)
        p = self.attn_scores(t)
        p = p.permute(0, 2, 1)
        u = torch.bmm(p, V_i)
        u = u.squeeze(1)

        return p, u

class Recognition_Module(nn.Module):
    def __init__(self):
        super(Recognition_Module, self).__init__()
        self.image_embeddings = ImageEmbedding()
        self.attention1 = Attention(dim_i=1024, dim_q=1024, dim_u=None)
        self.attention2 = Attention(dim_i=1024, dim_q=1024, dim_u=1024)

        self.classifier1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, prov_Num),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, alpha_Num),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, ad_Num),
        )

    def forward(self, x):
        embeddings = self.image_embeddings(x)
        p1, u1 = self.attention1(embeddings, None)
        p2, u2 = self.attention2(embeddings, u1)

        y0 = self.classifier1(u2)
        y1 = self.classifier2(u2)
        y2 = self.classifier3(u2)
        y3 = self.classifier4(u2)
        y4 = self.classifier5(u2)
        y5 = self.classifier6(u2)
        y6 = self.classifier7(u2)

        return [p1, p2], [y0, y1, y2, y3, y4, y5, y6]