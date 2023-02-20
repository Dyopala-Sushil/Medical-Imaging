from monai.networks.nets import UNet
from monai.networks.layers import Norm

def getModel(device):

    # Instantiating UNet model from MONAI 
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    return model