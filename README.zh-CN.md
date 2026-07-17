# ZF-ComfyUI-Helper

ComfyUI 输入与基础设置辅助节点。插件和节点统一使用 `ZF` 开头，方便在节点搜索里快速找到。

这个插件定位是轻量工具箱：以后遇到好用的尺寸、输入、基础设置、工作流辅助节点，可以慢慢加进来，不绑定某一个模型，也不依赖大型节点包。

## 节点

### ZF Resolution Selector

按比例、参考图、目标边长、目标百万像素和对齐规则计算宽高。

它不缩放图片，也不直接创建 latent，只输出干净的数值，方便连接到 `EmptyLatentImage`、采样器辅助节点、缩放节点或工作流设置节点。

支持：

- `original`：有参考图时使用参考图原始宽高；没接图时退回自定义宽高。
- `custom`：用自定义宽高作为比例和基础尺寸。
- 常见比例：`1:1`、`3:2`、`2:3`、`4:3`、`3:4`、`16:9`、`9:16`、`21:9`、`9:21`、`2:1`、`1:2`、`5:4`、`4:5`、`3:1`、`1:3`。
- 缩放模式：不缩放、按长边、按短边、按宽、按高、按百万像素。
- 对齐倍数：`1`、`8`、`16`、`32`、`64`、`128`。
- 对齐方式：就近、向下、向上。

### ZF Load Single Latent

从本地路径加载单个 `.latent`、`.safetensors` 或 `.sft` latent 文件。文件名可以不写后缀。

### ZF Load Latent Folders

从最多 10 个本地文件夹批量加载 latent 文件，每个文件夹对应一个输出。

### ZF Load Single Tensor Latent

从本地路径加载单个 `.pt` 或 `.pth` tensor latent。如果文件里是完整 latent 字典，会保留原结构；如果只有 tensor，会包装成 `{"samples": tensor}`。

### ZF Load Tensor Latent Folders

从最多 10 个本地文件夹批量加载 `.pt` 或 `.pth` tensor latent。

注意：`.pt` / `.pth` 依赖 PyTorch pickle 加载，只加载你自己生成或完全信任的文件。

### ZH Save Image

保存 PNG 图片到 ComfyUI 输出目录，但不写入 prompt、workflow 或其他 PNG 元数据。名字前面用 `ZH`，方便和 ComfyUI 原生 `Save Image` 区分。

### ZF 动态多路文本切换

用 `路线数量` 动态显示 1—32 路文本输入，节点中的数字按钮保持单路选择。点击一路后其余路线自动取消选择；减少路线导致原选择失效时，优先切到仍有连接的第一路。执行时若选中路线为空，节点会从上到下返回第一条非空文本，并通过 `实际路线` 输出最终采用的索引。

## 安装

放入 ComfyUI 的 `custom_nodes` 目录：

```bash
git clone https://github.com/Z-yaofang/ZF-ComfyUI-Helper.git
```

安装或更新后重启 ComfyUI。
