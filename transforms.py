from PIL import Image, ImageOps


class CenterSquareCropTransform:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        target = min(width, height)
        return ImageOps.fit(
            image,
            (target, target),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


