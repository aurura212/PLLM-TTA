# T-POP-2

## 实验目的

当两篇候选答案共享一个缺点时，用户选择 A 或 B 只能比较两者有差异的属性。T-POP-2 检验用户对已选答案的修改意见，能否提供原候选对缺失的监督，并在新问题上同时修复该缺点、保留用户选中答案的优点。

例如，A 有一个有用的例子，B 没有，两篇的结论都在最后。用户选 A 后补充「例子保留，以后先说结论」。选择监督有例子和无例子之间的排序。第二句额外监督结论位置。实验分为固定答案排序和本地模型自由生成两层。

## 数据和对照

`data/controlled_tasks.jsonl` 每行包含 `id`、`split`、`question`、`answer`、`explanation` 和 `example`。40 道英文解释题分为 16 道交互题、8 道开发题和 16 道测试题。开发题保留给规则检查和后续开发，每次正式汇总使用独立的测试题。每题通过调整结论位置和例子有无得到四种回答，四种回答使用同一结论和解释。

8 个模拟画像覆盖结论先行与靠后、有无例子，以及两个维度冲突时优先哪一个。每个画像的 16 个训练对只沿优先维度变化，另一维度保持在画像不偏好的值。训练对左右位置由种子决定。第 4、8、12、16 轮的同一次选择分别触发两个独立的可选问题，一个收集选择理由，另一个收集面向未来的修改意见。两类评价共享原始 A/B 和选择结果。

| 组别 | 使用的反馈 | 测试 |
|---|---|---|
| `choice` | 相同的二选一记录 | 固定排序、逐 token 生成 |
| `reason` | 二选一＋选择理由 | 固定排序、逐 token 生成 |
| `critique` | 二选一＋希望未来保留或修改的意见 | 固定排序、逐 token 生成 |
| `prompt` | 把原始 A/B、选择与同一组原始修改意见放入提示 | 同一基座直接生成 |

`fixed` 在第 0、4、8、16 轮记录三个奖励组。每道测试题有四类配对。`missing_only` 只改变训练中共享的维度，`observed_only` 只改变训练中有比较信号的维度，`tradeoff` 让两个维度互相冲突，`both_change` 同时改变两个维度。测试标签由模拟用户给出，训练只重放交互题的记录。`generate` 用第 16 轮权重在新题上自由生成。固定回放控制候选和选择差异，采用 T-POP 风格的逐 token 利用分支来测量这条反馈路径。

## 技术细节

### 奖励训练

用两个二值特征 \(\phi(y)=(\mathrm{lead},\mathrm{example})\) 表示已完成的回答，个人奖励为 \(r_u(y)=w_u^\top\phi(y)\)。把胜者放在差向量的前面，\(d=\phi(y_\mathrm{win})-\phi(y_\mathrm{lose})\)。选择损失是 \(-\log\sigma(w_u^\top d)\)。一个属性在两篇答案中相同时，它在这个差向量中的分量为零。

独立的解析调用接收题目、候选、选择和评价原文，提取 `attribute`、`direction`、`scope`、`evidence`。证据必须逐字出现在评价中。只有 `future` 且方向明确的条目进入长期奖励训练。相反方向的重复条目不能建立可靠约束。对于有效评价，构造 \(\delta\) 表示单属性从 0 到 1 或从 1 到 0 的变化。保持另一属性不动，增加 \(-\log\sigma(w_u^\top\delta)\) 的软约束。默认文本权重为 0.5，同一轮有多条约束时取平均。L2 正则系数为 1.0。三种奖励组对累积记录使用相同的两参数线性头和拟合器，原始文字与是否采纳逐条保存在反馈记录里。

### 前缀预测和生成

命令 `probe` 从同级 `T-POP-main/data/HelpSteer_train.json` 只读抽样，经 Unicode、大小写和空白归一化后与 40 道受控题去重。默认 72 道用于训练、24 道用于验证。冻结基座在每题采样四段续写，抽取含 `Answer:`、`Explanation:` 和可选 `Example:` 的有效完整回答。用最后一层的前缀隐藏状态做输入，完整回答的属性向量做标签，训练共享岭回归 \(g(h)\)。验证报告同时给出覆盖率和以训练集属性均值预测的 MSE 基线。

`generate` 使用同一个冻结模型和探针，把候选 token \(v\) 的分数设为 \(p_0(v\mid h)+\omega\sigma(w_u^\top g(hv))\)，在 top-8 中选最高分，默认 \(\omega=1\)、最多 128 个新 token。`prompt` 组用相同输出格式要求和生成长度，以原始反馈作为上下文。需要能容纳完整提示和输出预算的本地模型。模拟用户与裁判接收画像，评价解析器、本地模型及逐用户权重拟合器只接收各自可见的观察数据。

## 文件和运行方式

```text
T-POP-2/
├── README.md                   实验说明和复现步骤
├── requirements.txt            独立的 Python 依赖
├── data/
│   └── controlled_tasks.jsonl  40 道题的事实内容和划分
├── tpop2/
│   ├── __init__.py             Python 包入口
│   ├── __main__.py             命令行、配置检查和探针训练入口
│   ├── data.py                 题目加载、候选渲染、画像和去重抽样
│   ├── feedback.py             模拟用户、评价解析、盲评和 API 缓存
│   ├── reward.py               属性提取、反馈过滤和 Bradley–Terry 拟合
│   ├── decode.py               本地模型、共享探针和逐 token 选择
│   └── experiment.py           反馈回放、对照实验和汇总报告
└── tests/
    └── test_pipeline.py       离线集成测试和关键约束回归测试
```

固定答案和报告只依赖 NumPy、SciPy 和 Python 标准库。真实生成还需要 PyTorch、Transformers 和预先准备的本地 Hugging Face 因果语言模型权重。

### Linux 服务器

在本地工作区根目录执行下面的上传命令。将 `user@server` 换成你的 SSH 地址。这里把 HelpSteer 题库放进项目的 `data/` 目录，后续用 `--helpsteer` 指定它。

```bash
ssh user@server 'mkdir -p ~/research'
scp -r T-POP-2 user@server:~/research/
scp T-POP-main/data/HelpSteer_train.json user@server:~/research/T-POP-2/data/
```

登录服务器后，在分配到 GPU 的节点配置环境。以下示例使用 Python 3.12。先在激活的环境里执行与你的服务器兼容的 Linux CUDA 安装命令，再安装项目依赖。 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)

```bash
cd ~/research/T-POP-2
python3.12 -m venv .venv
source .venv/bin/activate
nvidia-smi
# 在这里执行 PyTorch 安装选择器给出的命令。
python -m pip install -r requirements.txt
python -B -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -B -m unittest discover -s tests -v
```

已有模型时，把 `tpop2_model` 指向服务器上的 Hugging Face 权重目录。首次联调可以用下面的 1.5B 指令模型。它用于检查流程，正式比较使用同一个选定模型的同一份权重。 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Hugging Face 下载命令](https://huggingface.co/docs/huggingface_hub/guides/cli#download-files)

```bash
tpop2_model="$HOME/models/Qwen2.5-1.5B-Instruct"
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$tpop2_model"
```

设置 API 环境变量。将模型 ID 占位符换成提供方的实际值。`TPOP2_BASE_URL` 默认为 `https://api.openai.com/v1`，使用其他兼容端点时将它设为提供方的 API 基址。若端点拒绝 `seed`、`temperature` 或 `response_format`，设置 `TPOP2_API_MODE=portable` 后重试。

```bash
read -rsp 'API key: ' TPOP2_API_KEY
printf '\n'
export TPOP2_API_KEY
export TPOP2_USER_MODEL='<模拟用户模型 ID>'
export TPOP2_JUDGE_MODEL='<不同的裁判模型 ID>'
```

先做 **1 个画像、2 道测试题**的联调。固定答案阶段在 CPU 上即可运行，探针和真实生成使用 GPU。两次模型命令使用相同的 `--model` 路径，生成会自动读取前一步的 `outputs/probe/probe.npz`。

```bash
python -B -m tpop2 fixed --seeds 17 --profile-limit 1 --test-limit 2 \
  --out-dir outputs/smoke-fixed
python -B -m tpop2 probe --model "$tpop2_model" --device cuda:0 \
  --helpsteer data/HelpSteer_train.json
python -B -m tpop2 generate --model "$tpop2_model" --device cuda:0 \
  --fixed-dir outputs/smoke-fixed --out-dir outputs/smoke-generation
python -B -m tpop2 report --run-dir outputs/smoke-fixed outputs/smoke-generation
```

联调通过后运行正式规模。模型权重与路径相同时复用探针。更换模型时重新训练探针。正式默认规模最多约 3,456 次 API 调用，实际次数随非空评价与缓存命中而变。

```bash
python -B -m tpop2 fixed
python -B -m tpop2 generate --model "$tpop2_model" --device cuda:0
python -B -m tpop2 report
```

### Windows PowerShell

从项目目录运行。固定答案和报告使用 CPU 即可。GPU 实验使用与设备匹配的 PyTorch 构建。

```powershell
Set-Location 'D:\USTC\科研\研究方向\PLLM\T-POP-2'
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B -m tpop2 --help
```

为 `fixed` 和 `generate` 配置兼容 Chat Completions 的端点。两种角色的模型 ID 必须不同。`TPOP2_PARSER_MODEL` 可选，默认与模拟用户模型相同，但使用新的调用且只看观察到的评价。`TPOP2_BASE_URL` 缺省为 `https://api.openai.com/v1`。`TPOP2_API_MODE=strict` 会请求 `seed`、零温度和 JSON 模式。兼容端点若不接受这些字段，可设置 `portable`，此时仍要求返回 JSON，复跑以缓存固定 API 抽样。密钥由环境变量传入。

```powershell
$env:TPOP2_API_KEY = Read-Host 'API key' -MaskInput
$env:TPOP2_USER_MODEL = '<模拟用户模型 ID>'
$env:TPOP2_JUDGE_MODEL = '<独立裁判模型 ID>'
$tpop2Model = '<本地模型权重目录>'

python -B -m tpop2 fixed
python -B -m tpop2 probe --model $tpop2Model
python -B -m tpop2 generate --model $tpop2Model
python -B -m tpop2 report
```

先做小样本联调时，可以只收集一个画像和两个新题的结果，仍保留完整的 16 轮交互。探针默认使用正式的 72/24 划分，小样本生成复用已经训练好的探针。用不同输出目录区分不同配置。

```powershell
python -B -m tpop2 fixed --seeds 17 --profile-limit 1 --test-limit 2 --out-dir outputs/smoke-fixed
python -B -m tpop2 generate --model $tpop2Model --fixed-dir outputs/smoke-fixed --out-dir outputs/smoke-generation
python -B -m tpop2 report --run-dir outputs/smoke-fixed outputs/smoke-generation
```

## 如何读结果

- `outputs/fixed/feedback.jsonl` 包含逐轮选择、原话、解析出的约束和是否涉及训练中共享的属性。解析结构或证据不合格时保留原话和 `parse_error`，该评价的个人奖励更新量为零。`fixed_predictions.jsonl` 保存评分的每个候选对。先看 `fixed_metrics.json` 中 `missing_only` 的第 16 轮结果和平分占比，再看 `mentions_missing` 与 `eligible_missing` 的差异。无法区分的配对计半分，并单列平分、正确和错误数。
- `outputs/probe/probe_metrics.json` 记录训练与验证题、有效前缀数和特征 MSE。若某个属性在训练时只有一个值，或者探针未优于均值基线，程序会提示。这种结果意味着生成实验需要结合探针质量解释。
- `outputs/generation/generations.jsonl` 保存四组生成文本，`generation_scores.jsonl` 保存独立裁判的匿名 A/B 判断。`generation_metrics.json` 和 `.csv` 汇总胜、平、负以及修正、保留、两者同时满足和正确性。修正以隐藏画像对共享维度的目标值为准。保留以训练中已选回答在变化维度上的多数值为准。选择没有多数值时，该画像的相应记录从保留与共同满足的分母中剔除。区间按模拟画像聚类自助，描述的是这个受控面板。
- `report` 从原始 JSONL 重新计算结果，无需模型或 API。成功的 API 原始 JSON 和解析值缓存于各阶段的 `api_cache/`。同配置重跑复用缓存。默认规模共有 384 次学习阶段选择、每类 96 次可选评价、1,536 次固定测试选择和 1,152 次自由生成盲评，非空评价还各需一次解析。不同提供方的冷跑输出可能变化，复现时保留 API 缓存、模型目录和运行清单。

固定答案阶段检验新增监督是否带来排序增益。自由生成阶段检验学进奖励头的修改意见能否优于直接记住同一句话。模拟阶段记录选择、评价次数和文字长度，真人时间成本需要后续实测。离线测试中注入的模拟 API 与小型测试解码器只验证程序和数据流。
