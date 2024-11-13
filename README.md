# DL-Compiler

---

## 前端v1.0

目前完成：
- 解析ONNX模型
- 常数折叠
- 单输入单输出的算子融合
- YOLOv5s残差结构融合
- Concat融合
- 检查算子需要存储、读取的张量
- 简易输出到 json

算子支持状态：

| 算子名       | 状态 | 算子名       | 状态 |
|-----------|:--:|-----------|:--:|
| Mul       | √  | Pow       | √  |
| Add       | √  | Sub       | √  |
| Div       | √  | Constant  | √  |
| Conv      | √  | Pool      | √  |
| Transpose | √  | Pad       | √  |
| ReLU      | √  | Sigmoid   | √  |
| Concat    | √  | Split     | √  |
| Reshape   | √  | Resize    | √  |
| Shape     | √  | Unsqueeze | √  |
| Floor     | √  | Slice     | √  |



## 后端

### 目前支持:

|   芯片   | 状态  |
|:------:|:---:|
| Ada200 | ing |

| 模块  | 状态  |
|:---:|:---:|
| PRE |  ×  |
| CIM | ing |
| VPU |  ×  |