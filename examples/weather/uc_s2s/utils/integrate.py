from torch import nn

class Integrator(nn.Module):
    def __init__(self, params, surface_ff_std, surface_delta_std, upper_air_ff_std, upper_air_delta_std):
        super().__init__() 
        self.atmo_resolution = [params.num_levels] + params.horizontal_resolution
        self.surface_ff_std = nn.parameter.Parameter(surface_ff_std, requires_grad = False)
        self.surface_delta_std = nn.parameter.Parameter(surface_delta_std, requires_grad = False)
        self.upper_air_ff_std = nn.parameter.Parameter(upper_air_ff_std, requires_grad = False)
        self.upper_air_delta_std = nn.parameter.Parameter(upper_air_delta_std, requires_grad = False)
        if hasattr(params, 'delta_integrator'):
            delta_integrator = params.delta_integrator
        else:
            delta_integrator = 'forward_euler'
        if delta_integrator == 'forward_euler':
            self.delta_integrator = forward_euler
        else:
            raise ValueError(f'delta_integrator must be in {["forward_euler"]}')
        
    def forward(self, surface, upper_air, surface_dx, upper_air_dx):
        output_surface = self.delta_integrator(surface, surface_dx * (self.surface_delta_std / self.surface_ff_std).reshape(1, -1, 1, 1), 1.)
        output_upper_air = self.delta_integrator(upper_air, 
                                                    upper_air_dx * (self.upper_air_delta_std / self.upper_air_ff_std).reshape(1, -1, self.atmo_resolution[0], 1, 1),
                                                    1.)
        return output_surface, output_upper_air

def forward_euler(x, dx, dt):
    return x + dx*dt