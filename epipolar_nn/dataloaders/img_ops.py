import PIL.ImageFilter
import PIL.Image


def pyrdown(image: PIL.Image.Image) -> PIL.Image.Image:
    """Implements OpenCV's pyrdown function with default arguments"""
    gauss_kernel = PIL.ImageFilter.Kernel(
        (5, 5),
        [
            1,  4,  6,  4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1,  4,  6,  4, 1
        ]
    )
    # Note the border mode does not match the default
    # BORDER_REFLECT_101 option in OpenCV. Instead,
    # the border pixels are simply copied into the returned
    # image, and only the inner pixels are defined by convolutions.
    low_pass_image = image.filter(gauss_kernel)

    # in order to remove the border artifact, we just remove the border
    low_pass_image = low_pass_image.crop((1, 1, low_pass_image.width - 1, low_pass_image.height - 1))

    # Next, we decimate the image. We want to remove roughly every other row and column.
    # This means we want to resize the image to half of its current dimensions.
    # If a dimension is odd, we round up (to match OpenCV pyrdown).
    # Right now the width and height are two less than the input to remove the border
    # artifact. Combined with the next line of code, we find the output dimensions.
    # Overall output_size = ((input_img.width - 1) // 2, (input_img.height - 1) // 2)
    target_size = ((low_pass_image.width + 1) // 2, (low_pass_image.height + 1) // 2)
    return low_pass_image.resize(target_size)
