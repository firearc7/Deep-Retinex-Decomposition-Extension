# RelightNet - Adjusts illumination map for low-light enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelightNet(nn.Module):
    # Illumination enhancement network with U-Net style architecture
    
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu = nn.ReLU()
        
        # Encoder
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        # Decoder with skip connections
        self.net2_deconv1_1 = nn.Conv2d(channel*2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel*2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel*2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')

        # Multi-scale feature fusion
        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        # Concatenate illumination and reflectance
        input_img = torch.cat((input_R, input_L), dim=1)
        
        # Encoding path
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))

        # Decoding path with skip connections
        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        # Multi-scale fusion
        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        
        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        
        return output
