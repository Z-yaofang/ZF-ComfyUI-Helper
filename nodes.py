import hashlib
import math
import os

import numpy as np
import safetensors.torch
import torch
from PIL import Image

import folder_paths


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

LATENT_EXTENSIONS = (".latent", ".safetensors", ".sft")
TENSOR_LATENT_EXTENSIONS = (".pt", ".pth")
DEFAULT_LATENT_FOLDER = r"C:\Users\94319\Desktop\Latent"
DEFAULT_LATENT_NAME = "001 (1)"
DEFAULT_TENSOR_LATENT_NAME = "ComfyPickle_latent_00001"
MAX_FOLDER_INPUTS = 10


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


def _normalize_path(path):
    path = os.path.expandvars(os.path.expanduser(str(path).strip().strip('"')))
    return os.path.abspath(path)


def _with_latent_extension(file_name):
    file_name = str(file_name).strip().strip('"')
    if not file_name:
        raise ValueError("Latent file name is empty.")
    if file_name.lower().endswith(LATENT_EXTENSIONS):
        return file_name
    return "{}.latent".format(file_name)


def _with_tensor_latent_extension(file_name):
    file_name = str(file_name).strip().strip('"')
    if not file_name:
        raise ValueError("Tensor latent file name is empty.")
    if file_name.lower().endswith(TENSOR_LATENT_EXTENSIONS):
        return file_name
    return "{}.pt".format(file_name)


def _resolve_latent_file(folder_path, file_name):
    folder_path = _normalize_path(folder_path)
    file_name = _with_latent_extension(file_name)
    return _normalize_path(os.path.join(folder_path, file_name))


def _resolve_tensor_latent_file(folder_path, file_name):
    folder_path = _normalize_path(folder_path)
    file_name = _with_tensor_latent_extension(file_name)
    return _normalize_path(os.path.join(folder_path, file_name))


def _load_latent_file(path):
    path = _normalize_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Latent file not found: {}".format(path))

    latent = safetensors.torch.load_file(path, device="cpu")
    if "latent_tensor" not in latent:
        raise ValueError("Latent file has no latent_tensor key: {}".format(path))

    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215

    return latent["latent_tensor"].float() * multiplier


def _load_tensor_latent_file(path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
    path = _normalize_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Tensor latent file not found: {}".format(path))

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if isinstance(data, dict):
        if "samples" in data:
            latent = dict(data)
            if hasattr(latent["samples"], "float"):
                latent["samples"] = latent["samples"].float()
            latent.setdefault("has_ref", wan_has_ref)
            latent.setdefault("drop_last", wan_drop_last)
            latent.setdefault("looped", wan_looped)
            return latent
        if "latent_tensor" in data:
            return {
                "samples": data["latent_tensor"].float(),
                "has_ref": wan_has_ref,
                "drop_last": wan_drop_last,
                "looped": wan_looped,
            }

    if hasattr(data, "float"):
        return {
            "samples": data.float(),
            "has_ref": wan_has_ref,
            "drop_last": wan_drop_last,
            "looped": wan_looped,
        }

    raise ValueError("Unsupported tensor latent content in: {}".format(path))


def _make_latent(path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
    return {
        "samples": _load_latent_file(path),
        "has_ref": wan_has_ref,
        "drop_last": wan_drop_last,
        "looped": wan_looped,
    }


def _make_tensor_latent(path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
    return _load_tensor_latent_file(path, wan_has_ref, wan_drop_last, wan_looped)


def _hash_file(path):
    path = _normalize_path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_latent_files(folder_path):
    folder_path = _normalize_path(folder_path)
    if not os.path.isdir(folder_path):
        raise NotADirectoryError("Latent folder not found: {}".format(folder_path))

    files = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and name.lower().endswith(LATENT_EXTENSIONS):
            files.append(path)
    return sorted(files, key=lambda value: value.lower())


def _list_tensor_latent_files(folder_path):
    folder_path = _normalize_path(folder_path)
    if not os.path.isdir(folder_path):
        raise NotADirectoryError("Tensor latent folder not found: {}".format(folder_path))

    files = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and name.lower().endswith(TENSOR_LATENT_EXTENSIONS):
            files.append(path)
    return sorted(files, key=lambda value: value.lower())


def _folder_slot_inputs():
    inputs = {}
    inputs["wan_has_ref"] = ("BOOLEAN", {"default": True})
    inputs["wan_drop_last"] = ("BOOLEAN", {"default": False})
    inputs["wan_looped"] = ("BOOLEAN", {"default": False})
    for index in range(1, MAX_FOLDER_INPUTS + 1):
        inputs["address_{}".format(index)] = (
            "STRING",
            {
                "default": DEFAULT_LATENT_FOLDER if index == 1 else "",
                "multiline": False,
            },
        )
    return inputs


def _tensor_folder_slot_inputs():
    inputs = {}
    inputs["wan_has_ref"] = ("BOOLEAN", {"default": True})
    inputs["wan_drop_last"] = ("BOOLEAN", {"default": False})
    inputs["wan_looped"] = ("BOOLEAN", {"default": False})
    for index in range(1, MAX_FOLDER_INPUTS + 1):
        inputs["address_{}".format(index)] = (
            "STRING",
            {
                "default": DEFAULT_LATENT_FOLDER if index == 1 else "",
                "multiline": False,
            },
        )
    return inputs


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


class LoadLatentFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": (
                    "STRING",
                    {
                        "default": DEFAULT_LATENT_NAME,
                        "multiline": False,
                    },
                ),
                "folder_path": (
                    "STRING",
                    {
                        "default": DEFAULT_LATENT_FOLDER,
                        "multiline": False,
                    },
                ),
                "wan_has_ref": ("BOOLEAN", {"default": True}),
                "wan_drop_last": ("BOOLEAN", {"default": False}),
                "wan_looped": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "file_path")
    FUNCTION = "load"
    CATEGORY = "ZF Helper/Latent"
    SEARCH_ALIASES = ["ZF load single latent", "load latent by path", "single latent"]

    def load(self, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        path = _resolve_latent_file(folder_path, file_name)
        samples = _make_latent(path, wan_has_ref, wan_drop_last, wan_looped)
        return (samples, path)

    @classmethod
    def IS_CHANGED(cls, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        try:
            path = _resolve_latent_file(folder_path, file_name)
            return "{}:{}:{}:{}".format(_hash_file(path), wan_has_ref, wan_drop_last, wan_looped)
        except Exception:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        try:
            path = _resolve_latent_file(folder_path, file_name)
        except Exception as error:
            return str(error)
        if not os.path.isfile(path):
            return "Invalid latent file path: {}".format(path)
        if not path.lower().endswith(LATENT_EXTENSIONS):
            return "Unsupported latent file extension: {}".format(path)
        return True


class LoadLatentsFromFolderPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _folder_slot_inputs()}

    RETURN_TYPES = ("LATENT",) * MAX_FOLDER_INPUTS
    RETURN_NAMES = tuple("latent_{}".format(index) for index in range(1, MAX_FOLDER_INPUTS + 1))
    FUNCTION = "load_batch"
    OUTPUT_IS_LIST = (True,) * MAX_FOLDER_INPUTS
    CATEGORY = "ZF Helper/Latent"
    SEARCH_ALIASES = ["ZF load latent folder", "batch load latent folders", "folder latent"]

    def load_batch(self, **kwargs):
        outputs = []
        has_folder = False
        wan_has_ref = kwargs.get("wan_has_ref", True)
        wan_drop_last = kwargs.get("wan_drop_last", False)
        wan_looped = kwargs.get("wan_looped", False)
        for index in range(1, MAX_FOLDER_INPUTS + 1):
            folder = kwargs.get("address_{}".format(index), "")
            if not str(folder).strip():
                outputs.append([])
                continue

            has_folder = True
            files = _list_latent_files(folder)
            if not files:
                raise FileNotFoundError("No latent files found in folder: {}".format(_normalize_path(folder)))
            outputs.append([
                _make_latent(path, wan_has_ref, wan_drop_last, wan_looped)
                for path in files
            ])

        if not has_folder:
            raise ValueError("At least one folder path is required.")
        return tuple(outputs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            digest = hashlib.sha256()
            digest.update(str(kwargs.get("wan_has_ref", True)).encode("ascii"))
            digest.update(str(kwargs.get("wan_drop_last", False)).encode("ascii"))
            digest.update(str(kwargs.get("wan_looped", False)).encode("ascii"))
            for index in range(1, MAX_FOLDER_INPUTS + 1):
                folder = kwargs.get("address_{}".format(index), "")
                if str(folder).strip():
                    for path in _list_latent_files(folder):
                        digest.update(path.encode("utf-8", errors="ignore"))
                        digest.update(_hash_file(path).encode("ascii"))
            return digest.hexdigest()
        except Exception:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        folders = [
            kwargs.get("address_{}".format(index), "")
            for index in range(1, MAX_FOLDER_INPUTS + 1)
        ]
        folders = [folder for folder in folders if str(folder).strip()]
        if not folders:
            return "At least one folder path is required."

        for folder in folders:
            path = _normalize_path(folder)
            if not os.path.isdir(path):
                return "Invalid latent folder path: {}".format(path)
            if not _list_latent_files(path):
                return "No latent files found in folder: {}".format(path)
        return True


class LoadTensorLatentFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": (
                    "STRING",
                    {
                        "default": DEFAULT_TENSOR_LATENT_NAME,
                        "multiline": False,
                    },
                ),
                "folder_path": (
                    "STRING",
                    {
                        "default": DEFAULT_LATENT_FOLDER,
                        "multiline": False,
                    },
                ),
                "wan_has_ref": ("BOOLEAN", {"default": True}),
                "wan_drop_last": ("BOOLEAN", {"default": False}),
                "wan_looped": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "file_path")
    FUNCTION = "load"
    CATEGORY = "ZF Helper/Latent"
    SEARCH_ALIASES = ["ZF load tensor latent", "load pt latent", "torch latent"]

    def load(self, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        path = _resolve_tensor_latent_file(folder_path, file_name)
        latent = _make_tensor_latent(path, wan_has_ref, wan_drop_last, wan_looped)
        return (latent, path)

    @classmethod
    def IS_CHANGED(cls, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        try:
            path = _resolve_tensor_latent_file(folder_path, file_name)
            return "{}:{}:{}:{}".format(_hash_file(path), wan_has_ref, wan_drop_last, wan_looped)
        except Exception:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, file_name, folder_path, wan_has_ref=True, wan_drop_last=False, wan_looped=False):
        try:
            path = _resolve_tensor_latent_file(folder_path, file_name)
        except Exception as error:
            return str(error)
        if not os.path.isfile(path):
            return "Invalid tensor latent file path: {}".format(path)
        if not path.lower().endswith(TENSOR_LATENT_EXTENSIONS):
            return "Unsupported tensor latent file extension: {}".format(path)
        return True


class LoadTensorLatentsFromFolderPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _tensor_folder_slot_inputs()}

    RETURN_TYPES = ("LATENT",) * MAX_FOLDER_INPUTS
    RETURN_NAMES = tuple("latent_{}".format(index) for index in range(1, MAX_FOLDER_INPUTS + 1))
    FUNCTION = "load_batch"
    OUTPUT_IS_LIST = (True,) * MAX_FOLDER_INPUTS
    CATEGORY = "ZF Helper/Latent"
    SEARCH_ALIASES = ["ZF load tensor latent folder", "batch load pt latent folders", "folder tensor latent"]

    def load_batch(self, **kwargs):
        outputs = []
        has_folder = False
        wan_has_ref = kwargs.get("wan_has_ref", True)
        wan_drop_last = kwargs.get("wan_drop_last", False)
        wan_looped = kwargs.get("wan_looped", False)
        for index in range(1, MAX_FOLDER_INPUTS + 1):
            folder = kwargs.get("address_{}".format(index), "")
            if not str(folder).strip():
                outputs.append([])
                continue

            has_folder = True
            files = _list_tensor_latent_files(folder)
            if not files:
                raise FileNotFoundError("No tensor latent files found in folder: {}".format(_normalize_path(folder)))
            outputs.append([
                _make_tensor_latent(path, wan_has_ref, wan_drop_last, wan_looped)
                for path in files
            ])

        if not has_folder:
            raise ValueError("At least one folder path is required.")
        return tuple(outputs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            digest = hashlib.sha256()
            digest.update(str(kwargs.get("wan_has_ref", True)).encode("ascii"))
            digest.update(str(kwargs.get("wan_drop_last", False)).encode("ascii"))
            digest.update(str(kwargs.get("wan_looped", False)).encode("ascii"))
            for index in range(1, MAX_FOLDER_INPUTS + 1):
                folder = kwargs.get("address_{}".format(index), "")
                if str(folder).strip():
                    for path in _list_tensor_latent_files(folder):
                        digest.update(path.encode("utf-8", errors="ignore"))
                        digest.update(_hash_file(path).encode("ascii"))
            return digest.hexdigest()
        except Exception:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        folders = [
            kwargs.get("address_{}".format(index), "")
            for index in range(1, MAX_FOLDER_INPUTS + 1)
        ]
        folders = [folder for folder in folders if str(folder).strip()]
        if not folders:
            return "At least one folder path is required."

        for folder in folders:
            path = _normalize_path(folder)
            if not os.path.isdir(path):
                return "Invalid tensor latent folder path: {}".format(path)
            if not _list_tensor_latent_files(path):
                return "No tensor latent files found in folder: {}".format(path)
        return True


class ZHSaveImageNoMetadata:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save without ComfyUI metadata."}),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "The prefix for saved PNG files. Supports ComfyUI filename formatting.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ZF Helper/Image"
    DESCRIPTION = "Saves images to the ComfyUI output directory without prompt or workflow metadata."
    SEARCH_ALIASES = ["ZH save image", "save image no metadata", "clean save image"]

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )
        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = "{}_{:05}_.png".format(filename_with_batch_num, counter)
            img.save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "ZFResolutionSelector": ZFResolutionSelector,
    "LoadLatentFromPath": LoadLatentFromPath,
    "LoadLatentsFromFolderPath": LoadLatentsFromFolderPath,
    "LoadTensorLatentFromPath": LoadTensorLatentFromPath,
    "LoadTensorLatentsFromFolderPath": LoadTensorLatentsFromFolderPath,
    "ZHSaveImageNoMetadata": ZHSaveImageNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZFResolutionSelector": "ZF Resolution Selector",
    "LoadLatentFromPath": "ZF Load Single Latent",
    "LoadLatentsFromFolderPath": "ZF Load Latent Folders",
    "LoadTensorLatentFromPath": "ZF Load Single Tensor Latent",
    "LoadTensorLatentsFromFolderPath": "ZF Load Tensor Latent Folders",
    "ZHSaveImageNoMetadata": "ZH Save Image",
}
