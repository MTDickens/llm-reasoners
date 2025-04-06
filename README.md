![logo](assets/reasoners_icon.png#pic_center)

<p align="center">
  <a href="https://www.llm-reasoners.net/">首页</a>
  |
  <a href="https://arxiv.org/abs/2404.05221">论文 (COLM2024)</a>
  |
  <a href="https://www.llm-reasoners.net/blog">博客</a>
  |
  <a href="https://discord.gg/PxDJby9W">Discord</a>
  |
  <a href="https://maitrix.org/">@Maitrix.org</a>
</p>

---

**LLM Reasoners** 是一个旨在利用先进算法增强大型语言模型（LLM）执行复杂推理能力的库。它提供：

-   **前沿的推理算法**

    该库提供了用于 LLM 推理的最新搜索算法，例如：

    -   [Reasoner Agent](example/ReasonerAgent-Web) ([Deng et al., 2025](https://reasoner-agent.maitrix.org/))
    -   [使用 PRM 进行推理时扩展](examples/Inference-Scaling-SGL/math500) ([Snell et al., 2024](https://arxiv.org/abs/2408.03314))
    -   [通过规划进行推理（Reasoning-via-Planning），MCTS](examples/RAP) ([Hao et al., 2023](https://arxiv.org/abs/2305.14992))
    -   [思维树（Tree-of-Thoughts），BFS](examples/ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))

    <details>
      <summary>(显示更多支持的算法)</summary>

    -   [StructChem](examples/StructChem) ([Ouyang et al., 2023](https://arxiv.org/abs/2311.09656))
    -   [思维链（Chain-of-thoughts）](examples/CoT) ([Wei et al., 2022](https://arxiv.org/abs/2201.11903))
    -   [从少到多提示（Least-to-most prompting）](examples/Least-to-most) ([Zhou et al., 2022](https://arxiv.org/abs/2205.10625))
    -   [思维树（Tree-of-Thoughts），DFS](examples/ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))
    -   [自评估引导解码（Self-Eval Guided Decoding），Beam Search](examples/Self-Eval) ([Xie et al., 2023](https://arxiv.org/abs/2305.00633))
    -   [Grace 解码](examples/Grace) ([Khalifa et al., 2023](https://arxiv.org/abs/2305.14934))
    -   [Eurus](examples/Eurus) ([Yuan et al., 2024](https://arxiv.org/abs/2404.02078))
    -   [PromptAgent](examples/PromptAgent) ([Wang et al., 2023](https://arxiv.org/abs/2310.16427))
    -   [DRPO](examples/DRPO) ([Singla et al., 2024](https://aclanthology.org/2024.emnlp-main.1220/))

    </details>

-   **直观的可视化与解释**：我们的库提供了一个[可视化工具](https://www.llm-reasoners.net/)，帮助用户理解推理过程。即使对于像蒙特卡洛树搜索（Monte-Carlo Tree Search）这样的复杂推理算法，用户也只需**一行 Python 代码**即可轻松诊断和理解其过程。请参阅教程 [notebook](demo.ipynb) 中的示例。

-   **高效的 LLM 推理**：我们的库通过集成高性能 LLM 推理框架 [SGLang](https://github.com/sgl-project/sglang) 来优化高级推理技术的性能，该框架支持结构化生成（请查看此 [帖子](https://x.com/MaitrixOrg/status/1885387184557199857) 和 [示例](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/Inference-Scaling-SGL/math500)）。我们还支持其他 LLM 后端，如 `huggingface transformers`、`OpenAI API`、`Exllama`、`fairscale`、`llama.cpp` 等。

-   **严谨的实现与可复现性**：我们在实现中优先考虑精确性和可靠性，确保我们的算法不仅是理论概念，而且是实用的工具。LLM Reasoners 中实现的所有方法都经过精心设计，以忠实于其原始表述和性能。它为我们在 COLM2024 上发表的推理算法[分析](https://arxiv.org/abs/2404.05221)提供了支持。

    <details>

    <summary> (可复现性示例) </summary>

    -   LLM Reasoners 已经过测试，可成功复现 [思维树（Tree-of-Thoughts）](https://arxiv.org/abs/2305.10601)、[引导解码（Guided Decoding）](https://arxiv.org/abs/2305.00633) 和 [GRACE 解码](https://arxiv.org/abs/2305.14934) 的官方实现性能。我们列出了其论文中报告的结果 / 从其官方仓库复现的结果以供参考 (†)。部分结果基于前 100 个样本的子集 (*)。

    <div align="center">

    |方法|基础 LLM|GSM8k|
    |--|--|--|
    |[引导解码](https://arxiv.org/abs/2305.00633)<sup>†</sup>|CodeX (PAL)|0.80|-|-|-|-|-|
    |引导解码|CodeX (PAL)|[0.83\*](examples/guided_gsm8k)|-|-|-|-|-|

    |方法|基础 LLM|Game of 24|
    |--|--|--|
    |[思维树](https://arxiv.org/abs/2305.10601)<sup>†</sup>|GPT-3.5-turbo|0.22|
    |思维树|GPT-3.5-turbo|[0.22](examples/tot_game24)|

    |方法|基础 LLM|GSM8k|
    |--|--|--|
    |[GRACE 解码](https://arxiv.org/abs/2305.14934)<sup>†</sup>|Flan-T5-Large (Fine-tuned)|0.34|-|-|-|-|-|
    |GRACE 解码| Flan-T5-Large (Fine-tuned)|[0.33\*](examples/grace_gsm8k)|-|-|-|-|-|
    </div>

    </details>

## 最新消息
-   2025年2月21日：我们集成了 Deepseek R1 ([示例](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/LongCoT_Search/ProsQA))。另请查看我们对 R1 搜索模式的分析 ([帖子](https://x.com/MaitrixOrg/status/1893017035753574799))。

-   2025年2月6日：很高兴推出 **ReasonerAgent** - 一个完全开源、开箱即用的智能体，可以在 Web 浏览器中进行研究 🧐 并回答您的查询。请查看此 [帖子](https://x.com/MaitrixOrg/status/1887584291087098063)，并在此处探索[代码](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/ReasonerAgent-Web)！
-   2025年1月31日：LLM Reasoners 已集成 [SGLang](https://github.com/sgl-project/sglang)。只需一行代码更改，即可享受 100 倍加速！像用于推理时扩展（inference-time scaling）的 PRM 引导搜索等新应用也已可用。在此 [帖子](https://x.com/MaitrixOrg/status/1885387184557199857) 中查看更多详情。
-   2024年12月20日：我们现在支持在 [BrowserGym](https://github.com/ServiceNow/BrowserGym) 的 Web 环境中使用规划算法（MCTS、DFS/BFS、Beam Search），查看 [README](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/browsergym) 尝试一下！

<details>

<summary>(显示更多新闻)</summary>

-   2024年11月13日：我们集成了 [DRPO](https://aclanthology.org/2024.emnlp-main.1220/)，一种在 EMNLP 2024 发表的免调优对齐方法 ([链接](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/DRPO))。

-   2024年7月10日：我们关于 [LLM Reasoners](https://arxiv.org/abs/2404.05221) 的论文已被 [COLM 2024](https://colmweb.org/index.html) 接收！
-   2024年6月24日：[PromptAgent](https://arxiv.org/abs/2310.16427) 已加入 LLM Reasoners！让它帮助您为您的任务写下一个超详细的提示 ([此处](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/PromptAgent))。
-   2024年5月14日：查看 [Eurus](https://arxiv.org/abs/2404.02078)，一套为推理优化的 LLM。借助 LLM Reasoners，Eurus-RM 可以轻松将 Llama-8B 在 GSM8k 上的性能从 0.49 提升到 0.73 📈 ([代码](examples/Eurus))。
-   2024年5月2日：我们集成了第一个用于科学推理的方法 [StructChem](https://arxiv.org/abs/2311.09656)！在此处查看 [代码](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/StructChem)。
-   2024年4月22日：我们集成了 [Llama-3](https://github.com/meta-llama/llama3)，并提供了额外的实用 API（例如，自定义 EOS 标记，计算似然）
-   **2024年4月8日：我们介绍 LLM Reasoners 的新[论文](assets/Reasoners.pdf)已发布！**
-   2024年3月29日：[Grace 解码](https://arxiv.org/abs/2305.14934) 已被整合！
-   2023年10月25日：关于 LLM Reasoners 可视化工具的[视频教程](https://www.youtube.com/watch?v=5QfOxtiw_ZU)已上线。

-   2023年10月23日：Reasoning-via-Planning（通过规划进行推理）已被 EMNLP 2023 接收！查看我们更新了结果和讨论的[论文](https://arxiv.org/abs/2305.14992)！
</details>

## 库介绍

![库结构](assets/figure2_reasoners_v5.png)

我们将 LLM 推理算法抽象为三个关键组件：*奖励函数 (reward function)*、*世界模型 (world model)* 和 *搜索算法 (search algorithm)*（参见我们[论文](https://arxiv.org/abs/2404.05221)中的表述），分别对应库中的三个类：<tt>SearchConfig</tt>、<tt>WorldModel</tt> 和 <tt>SearchAlgorithm</tt>。此外，还有用于支持其他模块的 <tt>LLM API</tt>、用于评估或调试推理算法的 <tt>Benchmark</tt>（基准测试）和 <tt>Visualization</tt>（可视化）（图中中部）。要为特定领域实现一个推理算法（一个 <tt>Reasoner</tt> 对象），用户可以继承 <tt>SearchConfig</tt> 和 <tt>WorldModel</tt> 类，并导入一个预先实现的 <tt>SearchAlgorithm</tt>。我们还展示了一个使用 LLM Reasoners 通过 RAP 解决 Blocksworld 问题的具体示例（图底部）。

## 快速导览
让我们看一下在 Blocksworld 问题上进行推理的代码。请注意，此代码为演示目的进行了简化（可运行的 notebook 请参见[此处](demo.ipynb)）。

第一步是定义世界模型：你需要在 `init_state` 中根据问题设置初始状态，在 `is_terminal` 中判断一个状态是否为终止状态，最重要的是，用 `step` 定义世界动态：
```python
from typing import NamedTuple
import utils
from reasoners import WorldModel, LanguageModel
import copy

BWState = str
BWAction = str

class BlocksWorldModel(WorldModel[BWState, BWAction]):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> BWState:
        # 从给定问题中提取初始状态
        # 例如："the red block is clear, the blue block is clear..."
        return BWState(utils.extract_init_state(self.example))

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        # 调用 LLM 预测状态转移
        state = copy.deepcopy(state)
        # 加载提示让 LLM 预测下一个状态
        # 例如："... I have that <state>, if I <action>, then ..."
        world_update_prompt = self.prompt["update"].replace("<state>", state).replace("<action>", action)
        world_output = self.base_model.generate([world_update_prompt],
                                    eos_token_id="\n", hide_input=True, temperature=0).text[0].strip()
        new_state = utils.process_new_state(world_output)
        # 到目前为止，我们得到了执行动作后的新状态
        # 下面的部分是为了加速奖励计算

        # 我们想检查已满足子目标的比例，并将其用作奖励的一部分
        # 因为我们已经预测了新状态，所以可以在这里方便地检查它
        goal_reached = utils.goal_check(utils.extract_goals(self.example, new_state))
        # 返回新状态和附加字典（将传递给奖励函数）
        return new_state, {"goal_reached": goal_reached}

    def is_terminal(self, state: BWState) -> bool:
        # 定义终止状态的条件以停止搜索
        # 例如：所有子目标都已满足
        if utils.goal_check(utils.extract_goals(self.example), state.blocks_state) == 1:
            return True
        return False
```
然后，需要考虑如何搜索最优的推理链。这涉及到使用 `get_actions` 获取给定状态下的动作空间，以及最重要的、作为推理指导的 `reward`（奖励）。对于蒙特卡洛树搜索（Monte-Carlo Tree Search），我们可以额外定义一个 `fast_reward` 来加速推演（roll-out）阶段。
```python
import utils
from world_model import BWState, BWAction
from reasoners import SearchConfig, LanguageModel
class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        # 一些用于计算快速奖励或奖励的参数（下面解释）
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def get_actions(self, state: BWState) -> list[BWAction]:
        # 使用基于规则的函数提取所有合法动作
        return utils.generate_all_actions(state)

    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        # 构建一个上下文学习提示（类似于思维链推理中使用的提示）
        inputs = self.prompt["icl"].replace("<init_state>", state)\
            .replace("<goals>", utils.extract_goals(self.example))
        # 在提示后连接一个候选动作，并测试其对数似然
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]
        # 奖励是直觉和目标满足度的组合
        # 在 fast_reward 中，我们跳过目标满足度的计算，使用默认值
        fast_reward = intuition * self.reward_alpha + self.goal_reward_default * (1 - self.reward_alpha)
        # 缓存一些信息供稍后奖励计算使用（将传递给 `reward` 函数）
        details = {'intuition': intuition}
        return fast_reward, details

    def reward(self, state: BWState, action: BWAction,
               intuition: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        # 注意 `intuition`（在 `fast_reward` 中缓存）和 `goal_reached`（在 `step` 中缓存）会自动作为参数传递给此奖励函数
        if goal_reached == 1:
            # 如果达到目标状态，我们将分配一个大的奖励
            goal_reward = self.goal_reached_reward
        else:
            # 否则根据已满足子目标的比例分配奖励
            goal_reward = goal_reached
        # 奖励是直觉和目标满足度的组合
        reward = intuition * self.reward_alpha + goal_reward * (1 - self.reward_alpha)
        # 返回奖励和一个附加字典（将保存在日志中供以后可视化）
        return reward, {'intuition': intuition, 'goal_reached': goal_reached}
```
现在，我们准备应用推理算法来解决问题：
```python
from reasoners.algorithm import MCTS
from reasoners.lm import LLaMAModel
from world_model import BlocksWorldModel
from search_config import BWConfig

llama_model = LLaMAModel(llama_ckpts, llama_size, max_batch_size=1)
with open(prompt_path) as f:
    prompt = json.load(f)
world_model = BlocksWorldModel(base_model=base_model, prompt=prompt)
config = BWConfig(base_model=llama_model, prompt=prompt)
# 保存每次迭代的历史记录以供可视化
search_algo = MCTS(output_trace_in_each_iter=True)
reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
for i, example in enumerate(dataset):
    algo_output = reasoner(example)
    # 将 MCTS 结果保存为 pickle 文件
    with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
        pickle.dump(algo_output, f)
```
最后，我们可以轻松地可视化推理过程：
```python
import pickle
from reasoners.visualization import visualize
with open("logs/bw_MCTS/xxx/algo_output/1.pkl", 'rb') as f:
    mcts_result = pickle.load(f)

from reasoners.visualization.tree_snapshot import NodeData, EdgeData # 修正：EdgeData 也需要导入
from reasoners.algorithm.mcts import MCTSNode

# 默认情况下，状态会与节点一起显示，奖励以及在 `SearchConfig.reward` 中保存的字典会与边一起显示。
# 我们也可以定义一个辅助函数来自定义我们想在可视化工具中看到的内容。
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else None,
                     "satisfied": n.fast_reward_details if n.fast_reward_details else "Not expanded"})
def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    # 假设 reward_details 存储在 MCTSNode 的 reward_details 属性中（或者通过其他方式获取）
    # 注意：原始代码示例中 MCTSNode 没有直接存储 reward_details，这里假设可以通过某种方式访问
    # 实际应用中可能需要调整 MCTS 实现或此工厂函数来访问所需数据
    reward_info = n.reward_details if hasattr(n, 'reward_details') else {} # 假设 reward_details 可访问
    intuition = reward_info.get('intuition', None) # 从 reward_details 获取 intuition
    return EdgeData({"reward": n.reward, "intuition": intuition})

visualize(mcts_result, node_data_factory=blocksworld_node_data_factory,
                       edge_data_factory=blocksworld_edge_data_factory)
```
然后会弹出一个可视化结果的 URL。该图形将是交互式的，看起来像我们[演示网站](https://llm-reasoners.net/)上显示的示例。
## 安装

请确保使用 Python 3.10 或更高版本。

```bash
conda create -n reasoners python=3.10
conda activate reasoners
```

### 通过 `pip` 安装

```bash
pip install llm-reasoners
```

### 从 GitHub 安装
（推荐，如果你想运行 GitHub 仓库中的示例）

```bash
git clone https://github.com/Ber666/llm-reasoners --recursive
cd llm-reasoners
pip install -e .
```
添加 `--recursive` 参数会自动克隆 exllama 和 LLM-Planning。请注意，某些可选模块可能需要其他依赖项。详情请参考错误信息。

## 引用
本项目是以下论文的扩展：
```bibtex
@inproceedings{hao2023reasoning,
  title={Reasoning with Language Model is Planning with World Model},
  author={Hao, Shibo and Gu, Yi and Ma, Haodi and Hong, Joshua and Wang, Zhen and Wang, Daisy and Hu, Zhiting},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={8154--8173},
  year={2023}
}
@article{hao2024llm,
  title={LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models},
  author={Hao, Shibo and Gu, Yi and Luo, Haotian and Liu, Tianyang and Shao, Xiyan and Wang, Xinyuan and Xie, Shuhua and Ma, Haodi and Samavedhi, Adithya and Gao, Qiyue and others},
  journal={arXiv preprint arXiv:2404.05221},
  year={2024}
}
```