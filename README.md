# fishing

这是一个面向 Windows 的《魔兽世界》钓鱼辅助脚本。

它的核心流程是：
- 监听按键 `1`，在你抛竿后延时截图
- 用视觉模型识别鱼漂位置
- 监听系统回放音频里的咬钩声音
- 在检测到咬钩后自动右键收杆
- 如果开启了自动循环，会在一段随机延时后再次抛竿

当前仓库已经自带训练好的默认模型，不需要你自己先训练。

## 1. 环境要求

- Windows 10/11
- Python `3.11` 到 `3.13` 之间更稳妥
- 一台可以正常运行游戏的电脑
- 如果你是英伟达显卡，仓库现在会优先尝试使用 CUDA 做 ONNX 推理

## 2. 获取代码

```bash
git clone https://github.com/ObservedObserver/fishing.git
cd fishing
```

## 3. 安装依赖

建议在虚拟环境里安装：

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

说明：
- 在 Windows 上，`requirements.txt` 会安装 `onnxruntime-gpu`
- 如果你的机器没有可用的 NVIDIA CUDA 环境，也可以手动改成安装 `onnxruntime`

## 4. 关键文件

- 配置文件：[config.yaml](/Users/observedobserver/Documents/GitHub/fishing/config.yaml)
- 主程序入口：[main.py](/Users/observedobserver/Documents/GitHub/fishing/main.py)
- 默认推理模型：[models/bobber.onnx](/Users/observedobserver/Documents/GitHub/fishing/models/bobber.onnx)

现在默认模型已经是当前仓库里训练效果最好的版本之一，直接运行即可。

## 5. 启动前需要改什么

主要看 [config.yaml](/Users/observedobserver/Documents/GitHub/fishing/config.yaml) 里的这几项：

### `vision.template_dir`

这里填你的模板图片目录。Windows 下建议改成你自己的真实路径，例如：

```yaml
vision:
  template_dir: "C:\\Users\\YourName\\Pictures\\bobber_templates"
```

如果你不打算用 template fallback，也可以先留空目录，但通常保留模板更稳。

### `audio`

常见要调的是：

```yaml
audio:
  backend: auto
  loopback_speaker_contains: null
  input_device: null
```

如果默认抓不到系统回放音频，可以先跑下面的诊断命令，再决定怎么填。

### `vision.enable_precast_cleanup`

当前默认是关闭：

```yaml
vision:
  enable_precast_cleanup: false
```

这是一个处理“上一轮鱼漂没清干净”这种 edge case 的兜底机制。默认关闭，只有你实际遇到这个问题时再打开。

## 6. 常用命令

所有命令都建议在仓库根目录执行。

### 下载/确认模型

```bash
python main.py --config config.yaml download-model
```

如果本地已经有 [models/bobber.onnx](/Users/observedobserver/Documents/GitHub/fishing/models/bobber.onnx)，这一步通常不会做额外事情。

### 测试音频逻辑

```bash
python main.py --config config.yaml test-audio
```

### 查看可用音频设备并做诊断

```bash
python main.py --config config.yaml audio-diagnose --seconds 15
```

### 做音频链路自检

```bash
python main.py --config config.yaml audio-selftest
```

### 只监听，不做鼠标动作

```bash
python main.py --config config.yaml listen-test --seconds 15
```

这个模式最适合先确认：
- 音频有没有抓到
- 视觉有没有检测到鱼漂
- 日志打印是不是正常

### 检测到目标后只移动鼠标，不点击

```bash
python main.py --config config.yaml mouse-test --seconds 15
```

这个模式适合验证：
- 识别框大致准不准
- 鼠标是不是会移动到正确位置

### 正式运行

```bash
python main.py --config config.yaml run
```

运行后：
- 按一次 `1`：激活循环
- 按 `ESC`：暂停循环

## 7. 推荐的首次启动流程

第一次拿到仓库，建议按下面顺序来：

1. 安装依赖
2. 修改 [config.yaml](/Users/observedobserver/Documents/GitHub/fishing/config.yaml) 里的模板路径
3. 跑 `audio-diagnose`
4. 跑 `audio-selftest`
5. 跑 `listen-test`
6. 跑 `mouse-test`
7. 最后再跑 `run`

这样最容易把问题定位清楚。

## 8. 运行时日志怎么看

常见日志含义：

- `[vision] ONNX providers: ...`
  - 表示当前 ONNX 实际用了哪些 provider
  - 如果是 NVIDIA 环境，理想情况会看到 `CUDAExecutionProvider`

- `[key] detected 1 ... schedule locate at +...`
  - 说明检测到你按了 `1`，开始等待视觉定位

- `[key-move] moved to ...`
  - 说明视觉已经找到鱼漂，并把鼠标移过去了

- `[audio-click] ...`
  - 说明检测到咬钩声音，开始执行收杆

- `[cast] scheduled in ...`
  - 说明进入自动下一轮抛竿的等待阶段

## 9. 常见问题

### 1. 运行后完全没有反应

先确认：
- 你是不是在仓库根目录执行命令
- `--config config.yaml` 指向的是你刚改过的配置文件
- 你是否已经按了一次 `1`

### 2. 抓不到咬钩声音

先跑：

```bash
python main.py --config config.yaml audio-diagnose --seconds 15
python main.py --config config.yaml audio-selftest
```

如果没有拿到正确的系统回放设备，优先检查：
- 声卡驱动
- Windows 声音输出设备
- `audio.loopback_speaker_contains`
- `audio.input_device`

### 3. 可以识别，但鼠标位置不对

优先跑：

```bash
python main.py --config config.yaml mouse-test --seconds 15
```

这样可以先确认是视觉问题，还是坐标映射问题。

### 4. Windows 上没有走 CUDA

启动后看日志里有没有：

```text
[vision] ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

如果没有，常见原因是：
- 没装好 `onnxruntime-gpu`
- CUDA / cuDNN 运行时不匹配
- 显卡驱动版本不对

## 10. 训练相关

如果你只是想运行脚本，这一节可以先跳过。

当前仓库里有训练脚本：
- [scripts/train_bobber.py](/Users/observedobserver/Documents/GitHub/fishing/scripts/train_bobber.py)
- [scripts/train_bobber_overfit.py](/Users/observedobserver/Documents/GitHub/fishing/scripts/train_bobber_overfit.py)

训练好的产物会同步到 [models](/Users/observedobserver/Documents/GitHub/fishing/models) 里，默认文件是：
- `bobber.best.pt`
- `bobber.last.pt`
- `bobber.onnx`

## 11. 现在最简单的启动方式

如果你只想尽快跑起来：

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py --config config.yaml run
```

然后：
- 进入游戏
- 确保模板路径已经配好
- 按一次 `1`
- 观察日志和鼠标行为
