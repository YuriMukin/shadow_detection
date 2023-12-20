from torchvision.models import vgg19
import torch.nn.functional as F
import torch.nn as nn


class myClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(512*49,  32*32)

    def forward(self, x):
        x = F.sigmoid(self.fc0(x))
        return x
    
def getVGGBased(unit):
    model_vgg19_based = vgg19()
    model_vgg19_based.classifier = myClassifier()
    model_vgg19_based.to(unit)
    return model_vgg19_based


    
class SSCDnet(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SSCDnet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.batch_normalization = nn.LayerNorm(2048)
       
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)
        self.threshold = nn.Threshold(0.5, 0)
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(1024)
        
        self.dconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(6, 6), stride = (1, 1))
        self.dconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(6, 6), stride = (1, 1))
        self.dconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(6, 6), stride = (1, 1))
        self.dconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(6, 6), stride = (1, 1))
        self.dconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride = (1, 1))
        self.dconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride = (1, 1))
        self.dconv7 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride = (1, 1))
        
        
    def forward(self,input):
        conv = self.conv1(input)
        conv = self.conv2(conv)
        batchnorm = self.relu(self.batchnorm1(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv3(maxpool)
        conv = self.conv4(conv)
        batchnorm = self.relu(self.batchnorm2(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv5(maxpool)
        conv = self.conv6(conv)
        batchnorm = self.relu(self.batchnorm3(conv))
        maxpool = self.maxpool(batchnorm)
        
        dconv = self.dconv1(maxpool)
        dconv = self.dconv2(dconv)
        batchnorm = self.relu(self.batchnorm2(dconv))
        dconv = self.dconv3(batchnorm)
        dconv = self.dconv4(dconv)
        batchnorm = self.relu(self.batchnorm1(dconv))
        dconv = self.dconv5(batchnorm)
        dconv = self.relu(self.dconv6(dconv))
        output = self.sigmoid(self.dconv7(dconv))

        return output
    
def getSSCDnet(unit):
    model_SSCDnet = SSCDnet(3*32*32, 32*32)
    model_SSCDnet.to(unit)
    return model_SSCDnet



class callorRestorer(nn.Module):
    def __init__(self):
        super(callorRestorer, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm64 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=(3, 3), stride = (1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride = (1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride = (1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride = (1, 1))

        self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride = (1, 1))
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride = (1, 1))
        self.dconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride = (1, 1))
        self.dconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3 , kernel_size=(1, 1), stride = (1, 1))
        
    def forward(self,input):
        conv = self.relu(self.conv1(input))
        batchnorm = self.batchnorm32(conv)
        conv = self.relu(self.conv2(batchnorm))
        batchnorm = self.batchnorm64(conv)
        conv = self.relu(self.conv3(batchnorm))
        
        dconv = self.relu(self.dconv1(conv))
        dconv = self.relu(self.dconv2(dconv))
        batchnorm = self.batchnorm32(dconv)
        dconv = self.relu(self.dconv3(batchnorm))
        dconv = self.sigmoid(self.dconv4(dconv))

        return dconv
    
def getCallorRestorer(unit):
    model_callorRestorer = callorRestorer()
    model_callorRestorer.to(unit)
    return model_callorRestorer