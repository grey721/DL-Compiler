# DL-Compiler
---
## 前端v1.0
### 目前进度:

- 模型张量信息加载
- 模型解析入口
- ElemWise、Constant加载
- Conv、Pool加载
- ReLU、Sigmoid加载
- Reshape、Transpose加载
- Concat、Split加载加载
- Pad、Resize加载

| 近期完成 |  计划   |  待完成  |
|:----:|:-----:|:-----:|
|  量化  | NPU张量 | NPU算子 |

算子支持状态：

| 算子名       |  状态  |
|-----------|:----:|
| ElemWise  |  √   |
| Conv      |  √   |
| Pool      |  √   |
| Constant  |  √   |
| Transpose |  √   |
| ReLU      |  √   |
| Sigmoid   |  √   |
| Concat    |  √   |
| Reshape   |  √   |
| Resize    |  √   |
| Pad       |  √   |
| Split     |  √   |

加入量化后的设想：
- 假如需要哈希值：
  1. 修改筛选机制
  2. AllTensorNames字典的键改用张量哈希值，内容仍然为当前的id，算子加载时，从量化配置获得张量哈希值
- 假如无需哈希值，张量量化参数在模型文件的op内 or 将量化参数改到OpBase的属性内
- 这里哈希值作用：区分同名张量在不同位置，因为量化方案可能不同？？


## 后端
### 目前进度:
|  已完成  |  计划  |  待完成  |
|:-----:|:----:|:-----:|
|       |      |       |