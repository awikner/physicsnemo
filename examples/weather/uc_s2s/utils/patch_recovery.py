import torch
from torch import nn


class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class PatchRecovery3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
    

# borrowed from 
#https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConvICNR_2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans, num_lat = None, polar_pad = True, grid_has_poles = False):
        super().__init__()
        self.img_size = img_size
        assert patch_size[0] == patch_size[1], 'mismatch'
        if polar_pad:
            self.pad_poles = PolarPad2d((1, 1), num_lat=num_lat, grid_has_poles=grid_has_poles)
        else:
            self.pad_poles = nn.ZeroPad2d((0, 0, 1, 1))
        self.pad_circular = nn.CircularPad2d((1, 1, 0, 0))
        self.conv = nn.Conv2d(in_chans, out_chans*patch_size[0]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x):
        x_padded = self.pad_poles(self.pad_circular(x))
        output = self.conv(x_padded)
        #print(output.shape)
        
        output = self.pixelshuffle(output)
        #print(output.shape)
        
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]
    
class SubPixelConvICNR_2D_wHead(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans, diagnostic_variables = 0, diagnostic_head = True, land_variables = 0, 
                 ocean_variables = 0, num_lat = None, polar_pad = True, grid_has_poles = False, hidden_dim = 96):
        super().__init__()
        self.img_size = img_size
        self.diagnostic_head = diagnostic_head
        self.diagnostic_variables = diagnostic_variables
        self.land_variables = land_variables
        self.ocean_variables = ocean_variables
        assert patch_size[0] == patch_size[1], 'mismatch'
        if polar_pad:
            self.pad_poles = PolarPad2d((1, 1), num_lat=num_lat, grid_has_poles=grid_has_poles)
            self.head = nn.Sequential(
                PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            if diagnostic_variables > 0 and diagnostic_head:
                self.diagnostic_head = nn.Sequential(
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, diagnostic_variables, kernel_size=1, stride=1, padding=0)
                )
            if land_variables > 0:
                self.land_head = nn.Sequential(
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.land_variables, kernel_size=1, stride=1, padding=0)
                )
            if ocean_variables > 0:
                self.ocean_head = nn.Sequential(
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.ocean_variables, kernel_size=1, stride=1, padding=0)
                )
        else:
            self.pad_poles = nn.ZeroPad2d((0, 0, 1, 1))
            self.head = nn.Sequential(
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            if diagnostic_variables > 0 and diagnostic_head:
                self.diagnostic_head = nn.Sequential(
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, diagnostic_variables, kernel_size=1, stride=1, padding=0)
                )
            if self.land_variables > 0:
                self.land_head = nn.Sequential(
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.land_variables, kernel_size=1, stride=1, padding=0)
                )
            if self.ocean_variables > 0:
                self.ocean_head = nn.Sequential(
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.ocean_variables, kernel_size=1, stride=1, padding=0)
                )

        self.pad_circular = nn.CircularPad2d((1, 1, 0, 0))
        self.conv = nn.Conv2d(in_chans, hidden_dim*patch_size[0]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x):
        x_padded = self.pad_poles(self.pad_circular(x))
        x_upsample = self.conv(x_padded)
        #print(output.shape)
        
        x_upsample = self.pixelshuffle(x_upsample)
        #print(output.shape)
        
        _, _, H, W = x_upsample.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        x_upsample = x_upsample[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]
        output = self.head(x_upsample)
        if self.diagnostic_variables > 0 and self.diagnostic_head:
            diagnostic_output = self.diagnostic_head(x_upsample)
            output = torch.cat([output, diagnostic_output], dim = 1)
        if self.land_variables > 0:
            land_output = self.land_head(x_upsample)
            output = torch.cat([output, land_output], dim = 1)
        if self.ocean_variables > 0:
            ocean_output = self.ocean_head(x_upsample)
            output = torch.cat([output, ocean_output], dim = 1)
        return output


class SubPixelConvICNR_3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans, padded_front = False, polar_pad = True, num_lat = None,
                 grid_has_poles = False):
        super().__init__()
        self.img_size = img_size
        self.padded_front = padded_front
        assert patch_size[1] == patch_size[2], 'mismatch'
        if polar_pad:
            self.pad_poles = PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles)
        else:
            self.pad_poles = nn.ZeroPad3d((0, 0, 1, 1, 0, 0))
        self.pad_circular = nn.CircularPad3d((1, 1, 0, 0, 0, 0))
        self.conv = nn.Conv2d(in_chans//2, out_chans*patch_size[1]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[1])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[1])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x: torch.Tensor):
        # first, split in dimension
        # print(x.shape)
        x_padded = self.pad_poles(self.pad_circular(x))
        x_padded = x_padded.reshape(x_padded.shape[0], x_padded.shape[1]//2, 2, *x_padded.shape[2:])
        x_padded = x_padded.flatten(2, 3)
        if not self.padded_front:
            x_padded = x_padded[:, :, 0:self.img_size[0]]
        else:
            x_padded = x_padded[:, :, 1:self.img_size[0]+1] # to make 13 vertical dims
        x_padded = x_padded.movedim(-3, 1).flatten(0, 1)
        output = self.conv(x_padded)
        output = self.pixelshuffle(output)
        output = output.reshape(-1, self.img_size[0], *output.shape[1:]).movedim(1, -3)
        
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
    

class SubPixelConvICNR_3D_wHead(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans, padded_front = False, polar_pad = True, num_lat = None,
                 grid_has_poles = False, hidden_dim = 96):
        super().__init__()
        self.img_size = img_size
        self.padded_front = padded_front
        assert patch_size[1] == patch_size[2], 'mismatch'
        if polar_pad:
            self.pad_poles = PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles)
            self.head = nn.Sequential(
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=1, padding=0),
                nn.GELU(),
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=1, padding=0),
                nn.GELU(),
                nn.Conv3d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            """
            self.head = nn.Sequential(
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.ZeroPad3d((0, 0, 0, 0, 1, 1)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.ZeroPad3d((0, 0, 0, 0, 1, 1)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.Conv3d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            """
        else:
            self.pad_poles = nn.ZeroPad3d((0, 0, 1, 1, 0, 0))
            self.head = nn.Sequential(
                nn.ZeroPad3d((0, 0, 1, 1, 0, 0)),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=1, padding=0),
                nn.GELU(),
                nn.ZeroPad3d((0, 0, 1, 1, 0, 0)),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=1, padding=0),
                nn.GELU(),
                nn.Conv3d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            """
            self.head = nn.Sequential(
                nn.ZeroPad3d((0, 0, 1, 1, 1, 1)),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.ZeroPad3d((0, 0, 1, 1, 1, 1)),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                nn.Conv3d(hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )
            """
        self.pad_circular = nn.CircularPad3d((1, 1, 0, 0, 0, 0))
        self.conv = nn.Conv2d(in_chans//2, hidden_dim*patch_size[1]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[1])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[1])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x: torch.Tensor):
        # first, split in dimension
        # print(x.shape)
        x_padded = self.pad_poles(self.pad_circular(x))
        x_padded = x_padded.reshape(x_padded.shape[0], x_padded.shape[1]//2, 2, *x_padded.shape[2:])
        x_padded = x_padded.flatten(2, 3)
        if not self.padded_front:
            x_padded = x_padded[:, :, 0:self.img_size[0]]
        else:
            x_padded = x_padded[:, :, 1:self.img_size[0]+1] # to make 13 vertical dims
        x_padded = x_padded.movedim(-3, 1).flatten(0, 1)
        x_upsample = self.conv(x_padded)
        x_upsample = self.pixelshuffle(x_upsample)
        x_upsample = x_upsample.reshape(-1, self.img_size[0], *x_upsample.shape[1:]).movedim(1, -3)
        
        _, _, Pl, Lat, Lon = x_upsample.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        x_upsample = x_upsample[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
        
        output = self.head(x_upsample)
        return output


        

    
class PatchRecovery5(nn.Module):
    ''' true upsampling with 3D conv
    '''
    def __init__(self, img_size, patch_size, num_levels, in_chans, out_chans, hidden_dim = 96, 
                 padded_front = False, polar_pad = True, num_lat = None,
                 grid_has_poles = False, downfactor = 4):
        #def __init__(self, 
        #             input_dim=None,
        #             dim=192,
        #             downfactor=4,
        #             hidden_dim=96,
        #             output_dim=69,
        #             n_level_variables=5):
        # input dim equals input_dim*z since we will be flattening stuff ?
        super().__init__()
        self.downfactor = downfactor
        self.num_levels = num_levels
        #if input_dim is None:
        #    input_dim = 8*dim

        self.img_size = img_size
        self.padded_front = padded_front
        assert patch_size[1] == patch_size[2], 'mismatch'
        if polar_pad:
            self.conv = nn.Sequential(
                PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0)),
                nn.Conv2d(in_chans//2, out_chans*patch_size[1]**2, kernel_size=3, stride=1, padding=0, bias=0),
                nn.PixelShuffle(patch_size[1])
            )
            nn.Conv2d(in_chans, num_levels*hidden_dim, kernel_size=1, stride=1, padding=0)
            self.head = nn.Sequential(
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU()
            )
            if downfactor == 4:
                self.head2 = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6, affine=True),
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
                nn.GELU(),
                PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles),
                nn.CircularPad3d((1, 1, 0, 0, 0, 0)),
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0), # kernel size 3 for interactions and smoothing
                nn.GELU())
            pad_poles_2D = PolarPad2d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles)
            pad_poles_3D = PolarPad3d((1, 1), num_lat = num_lat, grid_has_poles=grid_has_poles)
        else:
            self.input_conv = nn.Sequential(
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.Conv2d(in_chans, num_levels*hidden_dim, kernel_size=1, stride=1, padding=0)
            )
            pad_poles_2D = nn.ZeroPad2d((0, 0, 1, 1))
            pad_poles_3D = nn.ZeroPad2d((0, 0, 1, 1, 1, 1))
        pad_circular_2D = nn.CircularPad2d((1, 1, 0, 0))
        pad_circular_3D = nn.CircularPad3d((1, 1, 0, 0, 0, 0))
        self.input_conv = nn.Sequential(pad_circular_2D, 
                                        pad_poles_2D, 
                                        nn.Conv2d(in_chans, num_levels*hidden_dim, kernel_size=1, stride=1, padding=0))
        self.interp = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

        self.head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU(),
        )
        if downfactor == 4:
            self.head2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6, affine=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU())
        
        self.proj_surface = nn.Conv2d(hidden_dim, 4, kernel_size=1, stride=1, padding=0)
        self.proj_level = nn.Conv3d(hidden_dim, n_level_variables, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # recover enough levels
        bs = x.shape[0]
        x = x.flatten(1, 2) # put levels in the channel dim
        x = self.input_conv(x)
        x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
        x = self.interp(x)
        x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
        x = self.head(x)
        if self.downfactor == 4:
            x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
            x = self.interp(x)
            x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
            x = self.head2(x)

        output_surface = self.proj_surface(x[:, :, 0])
        output_level = self.proj_level(x[:, :, 1:])

        return output_level, output_surface.unsqueeze(-3)
    
class PolarPad2d(nn.Module):
    """
    Padding for convolutions on a 2D grid over the pole.

    Args:
        pad: (size of top padding, size of bottom padding)
        x: Image with shape (n_batches, n_channels, lat, lon)
    """
    def __init__(self, pad, num_lat = None, grid_has_poles = False):
        super().__init__()
        self.pad_top = pad[0]
        self.pad_bottom = pad[1]
        self.num_lat = num_lat if num_lat is not None else 64
        if not grid_has_poles:
            self.pad_idxs = torch.cat((torch.arange(self.pad_top), torch.arange(self.pad_top+1, self.num_lat+self.pad_top+1),
                                       torch.arange(self.num_lat+self.pad_top+2, self.num_lat+self.pad_top+self.pad_bottom+2))).long()
            self.pad_idxs.requires_grad_(requires_grad = False)

    def forward(self, x):
        try:
            assert x.shape[-2] == self.num_lat
        except:
            raise ValueError(f'Number of latitude grid points must equal {self.num_lat}.')
        try:
            assert x.shape[-1] % 2 == 0
        except:
            raise ValueError('Input to PolarPadding2D must have an even number of longitude grid points.')
        if not self.grid_has_poles:
            padded_x = nn.functional.pad(nn.functional.pad(x, (0, 0, 1, 1), mode = 'constant', value = 0.),
                                        (0, 0, self.pad_top, self.pad_bottom), mode = 'reflect')[..., self.pad_idxs, :]
        else:
            padded_x = nn.functional.pad(x, (0, 0, self.pad_top, self.pad_bottom), mode = 'reflect')
        padded_x[..., :self.pad_top, :] = torch.roll(padded_x[..., :self.pad_top, :], padded_x.shape[-1] // 2, dims = -1)
        padded_x[..., -self.pad_bottom:, :] = torch.roll(padded_x[..., -self.pad_bottom:, :], padded_x.shape[-1] // 2, dims = -1)
        return padded_x
    
class PolarPad3d(nn.Module):
    """
    Padding for convolutions on a 3D grid over the pole.

    Args:
        pad: (size of top padding, size of bottom padding)
        x: Image with shape (n_batches, n_channels, vertical_levels, lat, lon)
    """
    def __init__(self, pad, num_lat = 64, grid_has_poles = False):
        super().__init__()
        self.pad_top = pad[0]
        self.pad_bottom = pad[1]
        self.num_lat = num_lat if num_lat is not None else 64
        if not grid_has_poles:
            self.pad_idxs = torch.cat((torch.arange(self.pad_top), torch.arange(self.pad_top+1, self.num_lat+self.pad_top+1),
                                       torch.arange(self.num_lat+self.pad_top+2, self.num_lat+self.pad_top+self.pad_bottom+2))).long()
            self.pad_idxs.requires_grad_(requires_grad = False)

    def forward(self, x):
        try:
            assert x.shape[-2] == self.num_lat
        except:
            raise ValueError(f'Number of latitude grid points must equal {self.num_lat}.')
        try:
            assert x.shape[-1] % 2 == 0
        except:
            raise ValueError('Input to PolarPadding2D must have an even number of longitude grid points.')
        if not self.grid_has_poles:
            padded_x = nn.functional.pad(nn.functional.pad(x, (0, 0, 1, 1, 0, 0), mode = 'constant', value = 0.),
                                        (0, 0, self.pad_top, self.pad_bottom, 0, 0), mode = 'reflect')[..., self.pad_idxs, :]
        else:
            padded_x = nn.functional.pad(x, (0, 0, self.pad_top, self.pad_bottom, 0, 0), mode = 'reflect')
        padded_x[..., :self.pad_top, :] = torch.roll(padded_x[..., :self.pad_top, :], padded_x.shape[-1] // 2, dims = -1)
        padded_x[..., -self.pad_bottom:, :] = torch.roll(padded_x[..., -self.pad_bottom:, :], padded_x.shape[-1] // 2, dims = -1)
        return padded_x
    
class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
