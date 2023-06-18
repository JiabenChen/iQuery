import torch
import torch.nn as nn
import torch.nn.functional as F

class ANet(nn.Module):
    def __init__(self, init_weights=True):
        super(ANet, self).__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1,  64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = F.adaptive_max_pool2d(x, 1).view(b, 512)
        return x

class VGGish(nn.Module):
    def __init__(self, init_weights=True):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1,  64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = F.adaptive_max_pool2d(x, 1).view(b, 512)
        return x


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
            

class UnetIquery(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(UnetIquery, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.downrelu2 = nn.LeakyReLU(0.2, True)
        self.downrelu3 = nn.LeakyReLU(0.2, True)
        self.downrelu4 = nn.LeakyReLU(0.2, True)
        self.downrelu5 = nn.LeakyReLU(0.2, True)
        self.downrelu6 = nn.LeakyReLU(0.2, True)
        self.downrelu7 = nn.LeakyReLU(0.2, True)

        self.uprelu7 = nn.ReLU(True)
        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu6 = nn.ReLU(True)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu5 = nn.ReLU(True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu4 = nn.ReLU(True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu3 = nn.ReLU(True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu2 = nn.ReLU(True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uprelu1 = nn.ReLU(True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.use_bias = False

        self.downconv1 = nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downconv2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm2 = nn.BatchNorm2d(ngf*2)
        self.downconv3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm3 = nn.BatchNorm2d(ngf*4)
        self.downconv4 = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm4 = nn.BatchNorm2d(ngf*8)
        self.downconv5 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm5 = nn.BatchNorm2d(ngf*8)
        self.downconv6 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        self.downnorm6 = nn.BatchNorm2d(ngf*8)
        self.downconv7 = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        
        

        self.upconv7 = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm7 = nn.BatchNorm2d(ngf*8)
        self.upconv6 = nn.Conv2d(ngf*16, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm6 = nn.BatchNorm2d(ngf*8)
        self.upconv5 = nn.Conv2d(ngf*16, ngf*8, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm5 = nn.BatchNorm2d(ngf*8)
        self.upconv4 = nn.Conv2d(ngf*16, ngf*4, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm4 = nn.BatchNorm2d(ngf*4)
        self.upconv3 = nn.Conv2d(ngf*8, ngf*2, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm3 = nn.BatchNorm2d(ngf*2)
        self.upconv2 = nn.Conv2d(ngf*4, ngf, kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.upnorm2 = nn.BatchNorm2d(ngf)
        self.upconv1 = nn.Conv2d(ngf*2, fc_dim, kernel_size=3, stride=1, padding=1, bias=self.use_bias)

    def forward(self, x):
        x = self.bn0(x)
        #layer 1 down
        #outer_nc, inner_input_nc
        x1 = self.downconv1(x)
        #layer2 down
        
        x2 = self.downrelu2(x1)
        x2 = self.downconv2(x2)
        x2 = self.downnorm2(x2)
        
        #layer3 down
       
        x3 = self.downrelu3(x2)
        x3 = self.downconv3(x3)
        x3 = self.downnorm3(x3)
       
        #layer4 down
      
        x4 = self.downrelu4(x3)
        x4 = self.downconv4(x4)
        x4 = self.downnorm4(x4)
        
        #layer5 down:
        x5 = self.downrelu5(x4)
        x5 = self.downconv5(x5)
        x5 = self.downnorm5(x5)
        #layer6 down:
        x6 = self.downrelu6(x5)
        x6= self.downconv6(x6)
        x6 = self.downnorm6(x6)
        #layer7 down:
        x = self.downrelu7(x6)
        x = self.downconv7(x)
        x = self.uprelu7(x)
        x = self.upsample7(x)
        x = self.upconv7(x)
        x = self.upnorm7(x)
        

        #layer 6 up:
        x = self.uprelu6(torch.cat([x6, x], 1))
        x = self.upsample6(x)
        x = self.upconv6(x)
        x = self.upnorm6(x)
        

        #layer 5 up:
        x = self.uprelu5(torch.cat([x5, x], 1))
        x = self.upsample5(x)
        x = self.upconv5(x)
        x = self.upnorm5(x)
        
        #layer 4 up:
        x = self.uprelu4(torch.cat([x4, x], 1))
        x = self.upsample4(x)
        x = self.upconv4(x)
        x = self.upnorm4(x)
        x_latent = x 


        #layer3 up:
        x = self.uprelu3(torch.cat([x3, x], 1))
        x = self.upsample3(x)
        x = self.upconv3(x)
        x = self.upnorm3(x)

        #layer2 up:
        x = self.uprelu2(torch.cat([x2, x], 1))
        x = self.upsample2(x)
        x = self.upconv2(x)
        x = self.upnorm2(x)

        #layer 1 up:
        x = self.uprelu1(torch.cat([x1, x], 1))
        x = self.upsample1(x)
        x = self.upconv1(x)
        
        return x, x_latent