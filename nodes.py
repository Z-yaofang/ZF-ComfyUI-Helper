import math


ASPECT_RATIOS = [
    "original",
    "custom",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "16:9",
    "9:16",
    "21:9",
    "9:21",
    "2:1",
    "1:2",
    "5:4",
    "4:5",
    "3:1",
    "1:3",
]

SCALE_MODES = [
    "none",
    "longest",
    "shortest",
    "width",
    "height",
    "megapixels",
]

ROUND_MULTIPLES = ["1", "8", "16", "32", "64", "128"]
ROUND_MODES = ["nearest", "down", "up"]


def _image_size(image):
    if image is None:
        return 0, 0
    return int(image.shape[2]), int(image.shape[1])


def _parse_ratio(aspect_ratio, custom_width, custom_height, source_width, source_height):
    custom_width = max(1, int(custom_width))
    custom_height = max(1, int(custom_height))

    if aspect_ratio == "original":
        if source_width > 0 and source_height > 0:
            return source_width / float(source_height), source_width, source_height
        return custom_width / float(custom_height), custom_width, custom_height

    if aspect_ratio == "custom":
        return custom_width / float(custom_height), custom_width, custom_height

    if ":" in aspect_ratio:
        left, right = aspect_ratio.split(":", 1)
        try:
            rw = max(1, int(left.strip()))
            rh = max(1, int(right.strip()))
            return rw / float(rh), custom_width, custom_height
        except ValueError:
            pass

    return custom_width / float(custom_height), custom_width, custom_height


def _dims_from_longest(ratio, length):
    length = max(1, int(length))
    if ratio >= 1.0:
        return length, max(1, int(round(length / ratio)))
    return max(1, int(round(length * ratio))), length


def _dims_from_shortest(ratio, length):
    length = max(1, int(length))
    if ratio >= 1.0:
        return max(1, int(round(length * ratio))), length
    return length, max(1, int(round(length / ratio)))


def _dims_from_width(ratio, width):
    width = max(1, int(width))
    return width, max(1, int(round(width / ratio)))


def _dims_from_height(ratio, height):
    height = max(1, int(height))
    return max(1, int(round(height * ratio))), height


def _dims_from_megapixels(ratio, megapixels):
    pixels = max(1.0, float(megapixels) * 1_000_000.0)
    width = int(round(math.sqrt(pixels * ratio)))
    height = int(round(width / ratio))
    return max(1, width), max(1, height)


def _round_one(value, multiple, mode):
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    if multiple <= 1:
        return value
    if mode == "down":
        return max(multiple, (value // multiple) * multiple)
    if mode == "up":
        return max(multiple, int(math.ceil(value / float(multiple))) * multiple)
    return max(multiple, int(math.floor(value / float(multiple) + 0.5)) * multiple)


def _aspect_text(width, height):
    if width <= 0 or height <= 0:
        return "0:0"
    divisor = math.gcd(int(width), int(height))
    return "{}:{}".format(int(width) // divisor, int(height) // divisor)


class ZFResolutionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "original"}),
                "custom_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Used by custom ratio, and as fallback when original has no image input.",
                }),
                "custom_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Used by custom ratio, and as fallback when original has no image input.",
                }),
                "scale_mode": (SCALE_MODES, {"default": "none"}),
                "target_length": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Target side length for longest, shortest, width, or height modes.",
                }),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 256.0,
                    "step": 0.01,
                    "tooltip": "Target total megapixels when scale_mode is megapixels.",
                }),
                "round_to_multiple": (ROUND_MULTIPLES, {"default": "8"}),
                "round_mode": (ROUND_MODES, {"default": "nearest"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("width", "height", "megapixels", "summary", "source_width", "source_height")
    FUNCTION = "select_resolution"
    CATEGORY = "ZF Helper/Settings"

    def select_resolution(
        self,
        aspect_ratio,
        custom_width,
        custom_height,
        scale_mode,
        target_length,
        target_megapixels,
        round_to_multiple,
        round_mode,
        image=None,
    ):
        source_width, source_height = _image_size(image)
        ratio, base_width, base_height = _parse_ratio(
            aspect_ratio,
            custom_width,
            custom_height,
            source_width,
            source_height,
        )

        if scale_mode == "longest":
            width, height = _dims_from_longest(ratio, target_length)
        elif scale_mode == "shortest":
            width, height = _dims_from_shortest(ratio, target_length)
        elif scale_mode == "width":
            width, height = _dims_from_width(ratio, target_length)
        elif scale_mode == "height":
            width, height = _dims_from_height(ratio, target_length)
        elif scale_mode == "megapixels":
            width, height = _dims_from_megapixels(ratio, target_megapixels)
        else:
            if aspect_ratio in ("original", "custom"):
                width, height = int(base_width), int(base_height)
            else:
                base_length = max(int(base_width), int(base_height), 1)
                width, height = _dims_from_longest(ratio, base_length)

        multiple = int(round_to_multiple)
        width = _round_one(width, multiple, round_mode)
        height = _round_one(height, multiple, round_mode)

        megapixels = width * height / 1_000_000.0
        summary = "{}x{} | {} | {:.3f} MP | mode={} | round={} {}".format(
            width,
            height,
            _aspect_text(width, height),
            megapixels,
            scale_mode,
            round_mode,
            multiple,
        )
        return (width, height, float(megapixels), summary, source_width, source_height)


NODE_CLASS_MAPPINGS = {
    "ZFResolutionSelector": ZFResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZFResolutionSelector": "ZF Resolution Selector",
}
