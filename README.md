# DL-Compiler

---

## 前端v1.0

目前完成：
- 解析ONNX模型
- 消除常数算子
- 单输入单输出的算子融合
- YOLOv5残差结构融合
- 检查算子需要存储、读取的张量
- 简易输出到 json

算子支持状态：

| 算子名       | 状态 |
|-----------|:--:|
| ElemWise  | √  |
| Conv      | √  |
| Pool      | √  |
| Constant  | √  |
| Transpose | √  |
| ReLU      | √  |
| Sigmoid   | √  |
| Concat    | √  |
| Reshape   | √  |
| Resize    | √  |
| Pad       | √  |
| Split     | √  |
| Shape     | √  |
| Floor     | √  |
| Slice     | √  |
| Unsqueeze | √  |

## 后端

### 目前进度:

| 已完成 | 计划 | 待完成 |
|:---:|:--:|:---:|
|     |    |     |