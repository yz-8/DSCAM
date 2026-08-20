# Parser reliability audit guide

你只需要核查 parser 是否忠实抽取了模型回答，不需要重新看图判断视觉答案。

填写规则：

- `parser_ok = yes`：`parser_prediction` 与 `raw_prediction` 的真实短答案一致。
- `parser_ok = no`：parser 抽错、漏抽、多抽，或把模型回答归一化成了错误答案。
- `corrected_prediction`：只有 `parser_ok = no` 时填写，写成最短标准答案，例如 `red`、`leftmost`、`yes`、`no`。
- `notes`：可选，写 parser 错在哪里，例如 `raw says dark blue but parser extracted blue`。

不要按图片判断模型是否答对。比如 gold 是 `red`，raw 是 `blue`，parser 也抽成 `blue`，那么 parser 是对的，填 `yes`。

建议论文里报告：

- sampled 200 predictions across 5 models and 5 settings;
- report parser agreement rate;
- if parser errors are rare, state all main metrics are robust to parser audit.
