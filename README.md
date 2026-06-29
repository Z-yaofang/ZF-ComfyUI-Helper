# ZF-ComfyUI-Helper

Small, high-compatibility helper nodes for ComfyUI input and basic workflow settings.

This plugin is intended to be a lightweight toolbox for practical workflow setup tasks: resolution selection, input helpers, shared settings, and other small utility nodes that are useful across many models and workflows. Node names use the `ZF` prefix so they are easy to find in ComfyUI search.

## Nodes

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
