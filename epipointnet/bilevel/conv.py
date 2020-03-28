import torch
import torch.autograd

_f32_eps = float.fromhex('1p-23')


class Sobel2DArgMax(torch.autograd.Function):
    neighborhood_size = 5
    half_neighborhood_size = 2

    @staticmethod
    def forward(ctx, *args, **kwargs):
        bchw = args[0]

        bchw_shape = bchw.shape

        bcn = bchw.reshape((bchw_shape[0], bchw_shape[1], -1))

        linear_argmaxes = bcn.argmax(dim=2)  # shape: bxc

        bc_x = (linear_argmaxes % bchw_shape[3]).to(dtype=torch.float32)  # linear mod width
        bc_y = (linear_argmaxes // bchw_shape[3]).to(dtype=torch.float32)  # linear / width

        coords = torch.stack((bc_x, bc_y), dim=1)  # shape: bx2xc

        no_grad_due_to_border = (
                (bc_x < Sobel2DArgMax.half_neighborhood_size) |
                (bc_y < Sobel2DArgMax.half_neighborhood_size) |
                ((bchw.shape[3] - Sobel2DArgMax.half_neighborhood_size) <= bc_x) |
                ((bchw.shape[2] - Sobel2DArgMax.half_neighborhood_size) <= bc_y)
        )  # shape bxc

        ctx.save_for_backward(bchw, coords.clone(), no_grad_due_to_border)

        return coords

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_err_d_coords = grad_outputs[0]  # shape: bx2xc
        bchw, max_coords, no_grad_due_to_border = ctx.saved_tensors  # shapes: bxcxhxw, bx2xc
        max_coords = max_coords.to(torch.long)

        try:
            x_spatial_derivative_filter = ctx.BC2DAM_x_spatial_derivative_filter  # type: torch.Tensor
            if x_spatial_derivative_filter.device != bchw.device:
                x_spatial_derivative_filter = x_spatial_derivative_filter.to(device=bchw.device)
        except AttributeError:
            x_spatial_derivative_filter = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=bchw.dtype, device=bchw.device)

            # http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf
            # x_spatial_derivative_filter = torch.tensor([
            #     [-5, -4, 0, 4, 5],
            #     [-8, -10, 0, 10, 8],
            #     [-10, -20, 0, 20, 10],
            #     [-8, -10, 0, 10, 8],
            #     [-5, -4, 0, 4, 5]
            # ], dtype=bchw.dtype, device=bchw.device)

            ctx.BC2DAM_x_spatial_derivative_filter = x_spatial_derivative_filter

        return backward_conv_2d_arg_optim(d_err_d_coords, bchw, max_coords, no_grad_due_to_border,
                                          x_spatial_derivative_filter)


def backward_conv_2d_arg_optim(d_err_d_coords, bchw, max_coords, no_grad_due_to_border, x_spatial_derivative_filter):
    # Each column of the 2xc output represents the error gradient
    # w.r.t. the coordinates of the maximums in each of the corresponding channel maps.
    # Consider a single column and call it d_err_d_coords err(...).
    # d_err_d_coords err(...) is equivalent to [ dErr(...)/ du. dErr(...)/ dv ].T where
    # u and v are the x and y coordinates returned by the forward pass, respectively.

    # Consider a continuous function f which maps a 2d image and a coordinate
    # vector to an intensity. Call it f(x,y) where x is the image data
    # and y is the coordinate vector. f(x, y) can be thought of as an interpolant over
    # the input data x. Here, we treat the data contained in each b,c slice as a
    # 2d grid of samples of the continuous function f(x, y).

    # Next, let g(x) = argmax_y f(x,y). Gould, et al. show that the
    # gradient of g(x) w.r.t. x is equal to
    # -1 * (grad^2_{yy} f(x,g(x)))^-1  * [ d grad_y f(x,g(x))/ dx1, d grad_y f(x,g(x)/ dx2 ...,)

    # First, we find  grad^2_{yy} f(x, g(x)). Convolution operators can be used to approximate
    # the second order spatial derivatives.

    # Gather size and dimension information
    num_channels = bchw.shape[1]
    spatial_derivative_radii = (
        x_spatial_derivative_filter.shape[1] // 2,
        x_spatial_derivative_filter.shape[0] // 2
    )
    spatial_2nd_order_derivative_radii = (
        x_spatial_derivative_filter.shape[1] - 1,
        x_spatial_derivative_filter.shape[0] - 1
    )

    # Extract the patches used to calculate the gradient at the argmax
    max_coords_x = max_coords[:, 0, :]
    max_coords_y = max_coords[:, 1, :]
    argmax_patches = torch.zeros(
        bchw.shape[0], bchw.shape[1],
        2 * spatial_2nd_order_derivative_radii[1] + 1,
        2 * spatial_2nd_order_derivative_radii[0] + 1,
        dtype=bchw.dtype,
        device=bchw.device,
    )
    for b in range(bchw.shape[0]):
        for c in range(bchw.shape[1]):
            if not no_grad_due_to_border[b, c]:
                argmax_patches[b, c, :, :] = bchw[
                                             b,
                                             c,
                                             max_coords_y[b, c] - spatial_2nd_order_derivative_radii[1]:
                                             max_coords_y[b, c] + spatial_2nd_order_derivative_radii[1] + 1,
                                             max_coords_x[b, c] - spatial_2nd_order_derivative_radii[0]:
                                             max_coords_x[b, c] + spatial_2nd_order_derivative_radii[0] + 1
                                             ]  # shape: bxcx(2*2nd order radii + 1)x(2*2nd order radii + 1)

    x_spatial_derivative_weights = x_spatial_derivative_filter.expand((num_channels, 1, -1, -1))
    patches_du = torch.nn.functional.conv2d(argmax_patches, x_spatial_derivative_weights, groups=num_channels)

    y_spatial_derivative_weights = x_spatial_derivative_filter.t().expand((num_channels, 1, -1, -1))
    patches_dv = torch.nn.functional.conv2d(argmax_patches, y_spatial_derivative_weights, groups=num_channels)

    patches_dudu = torch.nn.functional.conv2d(patches_du, x_spatial_derivative_weights,
                                              groups=num_channels).squeeze(dim=-1).squeeze(dim=-1)  # shape: bxc

    patches_dudv = torch.nn.functional.conv2d(patches_du, y_spatial_derivative_weights,
                                              groups=num_channels).squeeze(dim=-1).squeeze(dim=-1)  # shape: bxc

    patches_dvdv = torch.nn.functional.conv2d(patches_dv, y_spatial_derivative_weights,
                                              groups=num_channels).squeeze(dim=-1).squeeze(dim=-1)  # shape: bxc

    bchw_grad_2_yy = torch.stack((
        torch.stack((patches_dudu, patches_dudv), dim=-1),
        torch.stack((patches_dudv, patches_dvdv), dim=-1)
    ), dim=-1)  # shape: bxcx2x2

    # Prevent the inverse from blowing up
    bchw_grad_2_yy[bchw_grad_2_yy.abs() < _f32_eps] = 0

    d_bchw_grad_y_dx = torch.zeros(
        (bchw.shape[0], bchw.shape[1], bchw.shape[2], bchw.shape[3], 2),
        dtype=bchw.dtype,
        device=bchw.device
    )  # shape:bxcxhxwx2

    # The convolution operator used to approximate the derivative is used to define
    # the mixed partial of f w.r.t. xy.

    for b in range(bchw.shape[0]):
        for c in range(bchw.shape[1]):
            if not no_grad_due_to_border[b, c]:
                d_bchw_grad_y_dx[
                b, c,
                max_coords_y[b, c] - spatial_derivative_radii[1]:
                max_coords_y[b, c] + spatial_derivative_radii[1] + 1,
                max_coords_x[b, c] - spatial_derivative_radii[0]:
                max_coords_x[b, c] + spatial_derivative_radii[0] + 1,
                :
                ] = torch.stack((
                    x_spatial_derivative_filter,
                    x_spatial_derivative_filter.t()),
                    dim=-1
                )

    # bchw_grad_2_yy_at_y.inverse(): bxcx2x2 -> bxcx1x1x2x2
    # d_bchw_grad_y_dx: bxcxhxwx2 -> bxcxhxwx2x1
    # This matmul seems to take longer than the convolutions. TODO: look into sparse representation
    coords_grad_x = -1 * torch.matmul(
        bchw_grad_2_yy.pinverse().unsqueeze(2).unsqueeze(2),
        d_bchw_grad_y_dx.unsqueeze(-1)
    )  # shape: bxcxhxwx2x1

    d_err_d_coords = d_err_d_coords.permute(0, 2, 1). \
        unsqueeze(2).unsqueeze(2).unsqueeze(2)  # shape: bx2xc-> bxcx2 -> bxcx1x1x1x2

    return torch.matmul(d_err_d_coords.to(dtype=coords_grad_x.dtype), coords_grad_x).squeeze(dim=-1).squeeze(dim=-1)
