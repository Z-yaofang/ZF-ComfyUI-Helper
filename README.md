# ZF-ComfyUI-Helper

Small, high-compatibility helper nodes for ComfyUI input and basic workflow settings.

This plugin is intended to be a lightweight toolbox for practical workflow setup tasks: resolution selection, input helpers, shared settings, and other small utility nodes that are useful across many models and workflows. Node names use the `ZF` prefix so they are easy to find in ComfyUI search.

## Nodes

### ZF EasySize Image

Resizes an optional image and mask using common model size presets, a custom size, or target longest/shortest side length.

Included preset groups:

- SD1.5
- SDXL
- FLUX
- WAN
- QWEN

Resize modes:

- `preset`: use one of the built-in model size presets.
- `custom`: use custom width and height.
- `longest_side`: preserve the input image or mask ratio and set the longest side.
- `shortest_side`: preserve the input image or mask ratio and set the shortest side.

Crop modes:

- `center_crop`: fill the target size and crop from the center.
- `stretch`: resize directly to the target size.

Outputs:

- `image`
- `mask`
- `width`
- `height`
- `summary`

### ZF EasySize Latent

Creates an empty latent from the same EasySize presets or a custom width and height.

Outputs:

- `latent`
- `width`
- `height`
- `summary`

### ZF EasySize Settings

Outputs width and height from the EasySize presets or a custom size without creating image or latent data.

Outputs:

- `width`
- `height`
- `summary`

### ZF Resolution Selector

Calculates width and height from a ratio, optional reference image, target side length, target megapixels, and rounding rules.

It does not resize images and does not create latent tensors. It only outputs clean numeric settings that can be connected to ComfyUI nodes such as `EmptyLatentImage`, sampler helpers, resize nodes, or workflow setting nodes.

Supported ratio modes:

- `original`: use the optional input image size and ratio. If no image is connected, fall back to custom width and height.
- `custom`: use custom width and custom height as the ratio and base size.
- Common fixed ratios such as `1:1`, `3:2`, `2:3`, `4:3`, `3:4`, `16:9`, `9:16`, `21:9`, `9:21`, `2:1`, `1:2`, `5:4`, `4:5`, `3:1`, and `1:3`.

Scale modes:

- `none`: keep the base size for `original` or `custom`; fixed ratios use the larger custom side as the longest side.
- `longest`: set the longest side to `target_length`.
- `shortest`: set the shortest side to `target_length`.
- `width`: set output width to `target_length`.
- `height`: set output height to `target_length`.
- `megapixels`: keep the selected ratio and solve width/height from `target_megapixels`.

Rounding:

- `round_to_multiple`: `1`, `8`, `16`, `32`, `64`, or `128`.
- `round_mode`: `nearest`, `down`, or `up`.

Outputs:

- `width`
- `height`
- `megapixels`
- `summary`
- `source_width`
- `source_height`

## Why It Exists

Resolution control is useful in far more than one model workflow. Keeping this as a small general helper plugin avoids depending on large node packs when all you need is stable width and height selection.

The first node is inspired by practical aspect-ratio selector workflows, but implemented as a clean standalone utility with no external dependencies.

### ZF Load Single Latent

Loads one safetensors-style latent file from a local folder path.

Supported extensions:

- `.latent`
- `.safetensors`
- `.sft`

The file name suffix is optional. For example, `001 (1)` resolves to `001 (1).latent`.

Outputs:

- `latent`
- `file_path`

### ZF Load Latent Folders

Loads latent files from up to 10 local folders. Each folder input has a matching latent-list output.

This is useful when you want to feed latent batches from fixed local paths without using browser upload dialogs or file picker nodes.

### ZF Load Single Tensor Latent

Loads one PyTorch tensor latent from a local folder path.

Supported extensions:

- `.pt`
- `.pth`

If the file contains a full ComfyUI latent dict with `samples`, it is preserved. If it contains only a tensor, the node wraps it as `{"samples": tensor}`.

### ZF Load Tensor Latent Folders

Loads `.pt` or `.pth` tensor latent files from up to 10 local folders.

Safety note: `.pt` and `.pth` files use PyTorch pickle loading. Only load tensor latent files that you created yourself or fully trust.

### ZH Save Image

Saves PNG images to the normal ComfyUI output directory without writing prompt, workflow, or extra PNG metadata.

It is intentionally named with `ZH` at the front so it is easy to distinguish from ComfyUI's built-in `Save Image` node while keeping similar behavior and output preview.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/Z-yaofang/ZF-ComfyUI-Helper.git
```

Restart ComfyUI after installation or updates.

## Localization

The default repository language and node definitions are English. A Simplified Chinese locale is included under `locales/zh/nodeDefs.json` for ComfyUI setups that load locale files.

## Notes

- This plugin has no model dependency.
- This plugin does not include workflows yet.
- The goal is to keep helper nodes small, predictable, and easy to reuse across different ComfyUI workflows.
- `ZF Load Single Tensor Latent` and `ZF Load Tensor Latent Folders` should only be used with trusted `.pt` / `.pth` files.
