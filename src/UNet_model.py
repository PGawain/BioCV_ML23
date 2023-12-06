from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch

def initialize_weights(self, m):
    ''' Initialize the weights of the network using He (Kaiming) initialization. '''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class conv_block(nn.Module):
    """ Convolution Block  """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
        # Initialization using He (Kaiming)
        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def initialize_weights(self, m):
        ''' Initialize the weights of the network using He (Kaiming) initialization. '''
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class up_conv(nn.Module):
    """ Up Convolution Block """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Initialization using He (Kaiming)
        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.up(x)
        return x
    
    def initialize_weights(self, m):
        ''' Initialize the weights of the network using He (Kaiming) initialization. '''
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class U_Net(nn.Module):
    """ UNet - Basic Implementation. Paper : https://arxiv.org/abs/1505.04597 """
    def __init__(self, in_ch=3, out_ch=5):
        super(U_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32, n1 * 64] 
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ------------- Added layers ----------------- #
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -------------------------------------------- #

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # ------------- Added layers ----------------- #
        self.Conv6 = conv_block(filters[4], filters[5])
        self.Conv7 = conv_block(filters[5], filters[6])

        self.Up7 = up_conv(filters[6], filters[5])
        self.Up_conv7 = conv_block(filters[6], filters[5])

        self.Up6 = up_conv(filters[5], filters[4])
        self.Up_conv6 = conv_block(filters[5], filters[4])
        # -------------------------------------------- #

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        #self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # ------------- Added layers ----------------- #
        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)

        e7 = self.Maxpool6(e6)
        e7 = self.Conv7(e7)

        d7 = self.Up7(e7)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)
        # -------------------------------------------- #

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        #out = self.Conv(d2)

        #d1 = self.active(out)
        output = torch.nn.functional.softmax(self.Conv(d2), dim=1)

        return output

def createModel(output_channels=5):
    model = U_Net(in_ch=1, out_ch=output_channels)
    return model

def summary(model, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Print a summary of the PyTorch model.
    
    Args:
    - model: PyTorch model
    - input_size: Tuple specifying the input size (e.g., (3, 224, 224) for an image with 3 channels and size 224x224)
    - device: Device on which to run the model ("cuda" or "cpu")
    """
    def forward_hook(module, input, output):
      num_params = sum(p.numel() for p in module.parameters())

      # Extract the first tensor from the OrderedDict
      if isinstance(output, dict):
          output_tensor = next(iter(output.values()))
      else:
          output_tensor = output

      print(f"{module.__class__.__name__.ljust(30)} | "
            f"Input shape: {str(input[0].shape).ljust(30)} | "
            f"Output shape: {str(output_tensor.shape).ljust(30)} | "
            f"Parameters: {num_params}")


    # Register the forward hook
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    # Move the model to the specified device
    model = model.to(device)
    input_size = (50, 288, 288)
    # Generate a dummy input tensor to trace the model
    input_tensor = torch.rand(*input_size).unsqueeze(0).to(device)
    
    # Print the header
    print("=" * 155)
    print("Layer (type)                   | Input Shape                             | Output Shape                            | Parameters")
    print("=" * 155)

    # Run a forward pass to get the summary
    model(input_tensor)

    # Remove the hooks
    for hook in hooks:
        hook.remove()
