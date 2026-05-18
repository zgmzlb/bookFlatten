# bookFlatten

一个使用经典计算机视觉实现的书页纠偏/展平小项目。

## 环境

- Python 3.12+
- `numpy`
- `opencv-python-headless`

安装依赖：

```bash
pip install -r requirements.txt
```

## 调用方式

```python
from correct_paper import correct_paper

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

result = correct_paper(image_bytes)

with open("image_corrected.jpg", "wb") as f:
    f.write(result)
```

## 实现思路

1. 将输入字节解码为 OpenCV 图像。
2. 将图片缩小到检测尺寸后，用 `grabCut` 从中心区域快速分离书页前景。
3. 对前景取最大外轮廓，计算凸包，再近似为四边形。
4. 根据四个角点进行透视变换，把书页拉正为矩形。
5. 如果 `grabCut` 失败，则回退到边缘检测 + 最大轮廓方案。

## 样例数据

仓库中的测试图片放在 `samples/` 目录。

任务书原始示例：

- `photo.jpg`
- `photo.png`
- `photo_corrected.jpg`
- `photo_corrected_1x1.jpg`

新增实拍样例：

- `1779114398540.jpg`
- `IMG_20260518_222309.jpg`
- `IMG_20260518_222317.jpg`
- `IMG_20260518_222335.jpg`
- `IMG_20260518_222558.jpg`
- `IMG_20260518_222601.jpg`

这些新增样例覆盖了几类情况：

- 跨页笔记本
- 倾斜拍摄的封面与内页
- 横竖方向混合的手机照片
- 更高分辨率的实拍输入

后续只要继续把测试图放进 `samples/`，批处理脚本就能自动扫描并处理。

## 本地演示

批量处理 `samples/` 中所有待测图片：

```bash
python demo.py
```

输出会写到 `output/batch_samples/`。

如果只处理单张图片：

```bash
python demo.py --input samples/photo.jpg --output-dir output
```

## 当前方案

- 检测阶段只在缩小图上进行，默认最长边压到 `550px`，降低耗时。
- 主路径使用 `grabCut` 从中心区域快速分离书页前景，适合“主体居中、背景相对简单”的场景。
- 分割结果取最大外轮廓，做凸包，再近似为四边形，尽量避免被书页内部内容干扰。
- 最后在原图分辨率上做透视变换，尽量保住文字清晰度。
- 如果分割失败，会回退到 `Canny + 形态学闭运算 + 最大轮廓` 的传统边缘方案。

## 本地耗时

使用 `samples/photo.jpg` 连续运行 20 次，测得：

- 平均耗时：`0.1573s`
- 最大耗时：`0.1863s`

对当前这组样例，满足任务书中“均摊 <= 0.4s、单次 <= 1s”的目标。
