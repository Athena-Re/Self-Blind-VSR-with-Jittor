import jittor as jt
import jittor.nn as nn
import math
import numpy as np


def make_model(args):
    pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    return Flow_PWC(pretrain_fn=pretrain_fn)


class Flow_PWC(nn.Module):
    def __init__(self, pretrain_fn=''):
        super(Flow_PWC, self).__init__()
        self.moduleNetwork = Network()
        print("Creating Flow PWC")

        # Note: Jittor doesn't directly support loading PyTorch models
        # You would need to convert the pretrained model or train from scratch
        if pretrain_fn != '.' and pretrain_fn.endswith('.pkl'):
            self.moduleNetwork.load_state_dict(jt.load(pretrain_fn))
            print('Loading Flow PWC pretrain model from {}'.format(pretrain_fn))

    def estimate_flow(self, tensorFirst, tensorSecond):
        b, c, intHeight, intWidth = tensorFirst.size()

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = nn.interpolate(tensorFirst,
                                                size=(intPreprocessedHeight, intPreprocessedWidth),
                                                mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = nn.interpolate(tensorSecond,
                                                 size=(intPreprocessedHeight, intPreprocessedWidth),
                                                 mode='bilinear', align_corners=False)

        outputFlow = self.moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)

        tensorFlow = 20.0 * nn.interpolate(outputFlow, size=(intHeight, intWidth),
                                          mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = jt.arange(0, W).view(1, -1).repeat(H, 1)
        yy = jt.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = jt.concat((xx, yy), 1).float()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.grid_sample(x, vgrid, padding_mode='border', align_corners=True)
        mask = jt.ones(x.size())
        mask = nn.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output, mask

    def execute(self, frame_1, frame_2):
        # flow
        flow = self.estimate_flow(frame_1, frame_2)
        return flow


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleTwo = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleThr = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleFou = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleFiv = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleSix = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

            def execute(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]

        class Decoder(nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

                if intLevel < 6:
                    self.moduleUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                    self.moduleUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                    self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.moduleOne = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleTwo = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleThr = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent + 256, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleFou = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent + 352, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleFiv = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent + 416, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1)
                )

                self.moduleSix = nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent + 448, out_channels=2, kernel_size=3, stride=1, padding=1)
                )

            def execute(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None

                    tensorVolume = nn.leaky_relu(FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond), scale=0.1)

                    tensorFeat = jt.concat([tensorVolume], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                    tensorVolume = nn.leaky_relu(FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)), scale=0.1)

                    tensorFeat = jt.concat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                tensorFeat = jt.concat([self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = jt.concat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = jt.concat([self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = jt.concat([self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = jt.concat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }

        class Refiner(nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = nn.Sequential(
                    nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(scale=0.1),
                    nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )

            def execute(self, tensorInput):
                return self.moduleMain(tensorInput)

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

    def execute(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])


# Jittor implementation of correlation function
def FunctionCorrelation(tensorFirst, tensorSecond):
    """
    Jittor implementation of correlation operation
    Returns correlation volume with shape [B, 81, H, W]
    """
    # Get input dimensions
    batch_size, channels, height, width = tensorFirst.size()
    
    # Pad inputs
    tensorFirst_padded = nn.pad(tensorFirst, [4, 4, 4, 4], mode='constant', value=0)
    tensorSecond_padded = nn.pad(tensorSecond, [4, 4, 4, 4], mode='constant', value=0)
    
    # Initialize output
    output = jt.zeros(batch_size, 81, height, width, dtype=tensorFirst.dtype)
    
    # Compute correlation for each displacement
    for i, dy in enumerate(range(-4, 5)):
        for j, dx in enumerate(range(-4, 5)):
            # Calculate displacement index
            disp_idx = i * 9 + j
            
            # Extract patches
            tensorSecond_shifted = tensorSecond_padded[:, :, 4+dy:4+dy+height, 4+dx:4+dx+width]
            
            # Compute correlation
            correlation = jt.sum(tensorFirst * tensorSecond_shifted, dim=1, keepdim=True)
            correlation = correlation / channels  # Normalize by number of channels
            
            output[:, disp_idx:disp_idx+1, :, :] = correlation
    
    return output


def Backward(tensorInput, tensorFlow):
    # Simplified backward warping
    B, C, H, W = tensorInput.size()
    
    # Create coordinate grids
    xx = jt.arange(0, W).view(1, -1).repeat(H, 1)
    yy = jt.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = jt.concat([xx, yy], 1).float()
    
    # Apply flow
    vgrid = grid + tensorFlow
    
    # Normalize to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.grid_sample(tensorInput, vgrid, padding_mode='zeros', align_corners=True)
    
    return output 