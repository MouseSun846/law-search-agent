import os
import json
import asyncio
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.file_management import ListDirectoryTool, ReadFileTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from deepagents.middleware.subagents import SubAgentMiddleware  
from langchain.agents.middleware import TodoListMiddleware
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig  
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver  
from typing import TypedDict, Annotated  
from langgraph.graph.message import add_messages  

os.environ["OPENAI_API_BASE"] = "http://10.1.30.1:18080/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

# 法律文件检索路径限制
LAW_DOCS_PATH = r"D:\Code\mcp\law\just-laws\docs\civil-and-commercial"


# --- 文件检索工具配置 ---

# 共用的路径规范化函数
def normalize_path(file_path: str, root_dir: str) -> str:
    """规范化路径，修正常见的路径格式错误"""
    # 去除首尾空格
    file_path = file_path.strip()
    
    # 如果是绝对路径，尝试提取相对部分
    if os.path.isabs(file_path):
        # 尝试提取根目录之后的相对路径
        abs_root = os.path.abspath(root_dir)
        abs_input = os.path.abspath(file_path)
        if abs_input.startswith(abs_root):
            # 提取相对路径
            file_path = os.path.relpath(abs_input, abs_root)
        else:
            # 尝试从路径中提取可能的相对部分
            parts = file_path.replace('\\', '/').split('/')
            # 查找 civil-and-commercial 之后的部分
            try:
                idx = parts.index('civil-and-commercial')
                file_path = '/'.join(parts[idx + 1:])
            except ValueError:
                pass
    
    # 去除前导斜杠
    while file_path.startswith('/') or file_path.startswith('\\'):
        file_path = file_path[1:]
    
    # 统一使用正斜杠
    file_path = file_path.replace('\\', '/')
    
    return file_path


def split_content_into_chunks(content: str, max_chunk_lines: int = 500) -> list[str]:
    """
    将文件内容按段落智能分块
    
    策略：
    1. 按双换行符（段落分隔）分割内容
    2. 每个分块最多包含 max_chunk_lines 行
    3. 尽量保持段落完整性，不在段落中间截断
    """
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_line_count = 0
    
    for line in lines:
        current_chunk.append(line)
        current_line_count += 1
        
        # 如果遇到空行（段落结束）且已经积累了足够的内容
        if line.strip() == '' and current_line_count >= max_chunk_lines:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_line_count = 0
        # 如果行数超过限制较多（1.5倍），强制分块
        elif current_line_count >= max_chunk_lines * 1.5:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_line_count = 0
    
    # 添加最后剩余的内容
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


# 工具1：获取文件分块信息
class GetFileChunksInfoInput(BaseModel):
    """获取文件分块信息的输入参数"""
    file_path: str = Field(description="要获取信息的文件路径（相对于法律文档根目录）")

class GetFileChunksInfoTool(BaseTool):
    """获取文件的分块信息（不读取实际内容）"""
    name: str = "get_file_chunks_info"
    description: str = """获取文件的分块信息，包括总块数、文件大小等元数据。
    
    参数：
    - file_path: 文件路径（相对于法律文档根目录）
    
    返回：文件的分块元数据（总块数、总行数、文件大小等），但不包含实际内容。
    使用此工具先了解文件结构，再使用 read_file_chunk 读取具体分块。"""
    args_schema: type[BaseModel] = GetFileChunksInfoInput
    root_dir: str = LAW_DOCS_PATH
    
    def _run(self, file_path: str) -> str:
        """获取文件分块信息"""
        try:
            # 规范化路径
            file_path = normalize_path(file_path, self.root_dir)
            
            # 构建完整路径
            full_path = os.path.join(self.root_dir, file_path)
            
            # 安全检查
            abs_root = os.path.abspath(self.root_dir)
            abs_path = os.path.abspath(full_path)
            if not abs_path.startswith(abs_root):
                return json.dumps({"error": "访问被拒绝：文件路径超出允许范围"}, ensure_ascii=False)
            
            if not os.path.exists(full_path):
                return json.dumps({"error": f"文件不存在: {file_path}"}, ensure_ascii=False)
            
            if os.path.isdir(full_path):
                return json.dumps({
                    "error": f"路径 '{file_path}' 是一个目录，不是文件",
                    "suggestion": "请指定目录中的具体文件，或使用 get_all_files 查看所有文件"
                }, ensure_ascii=False)
            
            # 读取文件内容并分块
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = split_content_into_chunks(content)
            
            # 计算统计信息
            total_lines = content.count('\n') + 1
            file_size_bytes = os.path.getsize(full_path)
            
            # 生成每个分块的概要信息
            chunk_summaries = []
            for idx, chunk in enumerate(chunks):
                chunk_lines = chunk.count('\n') + 1
                # 提取分块的前几行作为预览
                preview_lines = chunk.split('\n')[:3]
                preview = '\n'.join(preview_lines)
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                
                chunk_summaries.append({
                    "chunk_index": idx,
                    "lines": chunk_lines,
                    "preview": preview
                })
            
            return json.dumps({
                "file_path": file_path,
                "total_chunks": len(chunks),
                "total_lines": total_lines,
                "file_size_bytes": file_size_bytes,
                "chunk_summaries": chunk_summaries
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"获取文件信息时发生错误: {str(e)}"}, ensure_ascii=False)
    
    async def _arun(self, file_path: str) -> str:
        """异步执行"""
        return await asyncio.to_thread(self._run, file_path)


# 工具2：读取指定分块的内容
class ReadFileChunkInput(BaseModel):
    """读取文件分块的输入参数"""
    file_path: str = Field(description="要读取的文件路径（相对于法律文档根目录）")
    chunk_index: int = Field(description="要读取的分块索引（从0开始）")

class ReadFileChunkTool(BaseTool):
    """读取文件的指定分块内容"""
    name: str = "read_file_chunk"
    description: str = """读取文件的指定分块内容。
    
    参数：
    - file_path: 文件路径（相对于法律文档根目录）
    - chunk_index: 分块索引（从0开始）
    
    返回：指定分块的完整内容及其元数据（当前块索引、是否最后一块等）。
    使用前请先调用 get_file_chunks_info 了解文件有多少块。"""
    args_schema: type[BaseModel] = ReadFileChunkInput
    root_dir: str = LAW_DOCS_PATH
    
    def _run(self, file_path: str, chunk_index: int) -> str:
        """读取指定分块"""
        try:
            # 规范化路径
            file_path = normalize_path(file_path, self.root_dir)
            
            # 构建完整路径
            full_path = os.path.join(self.root_dir, file_path)
            
            # 安全检查
            abs_root = os.path.abspath(self.root_dir)
            abs_path = os.path.abspath(full_path)
            if not abs_path.startswith(abs_root):
                return json.dumps({"error": "访问被拒绝：文件路径超出允许范围"}, ensure_ascii=False)
            
            if not os.path.exists(full_path):
                return json.dumps({"error": f"文件不存在: {file_path}"}, ensure_ascii=False)
            
            if os.path.isdir(full_path):
                return json.dumps({
                    "error": f"路径 '{file_path}' 是一个目录，不是文件"
                }, ensure_ascii=False)
            
            # 读取文件并分块
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = split_content_into_chunks(content)
            
            # 验证索引
            if chunk_index < 0 or chunk_index >= len(chunks):
                return json.dumps({
                    "error": f"分块索引 {chunk_index} 超出范围（总共 {len(chunks)} 块，索引范围 0-{len(chunks)-1}）"
                }, ensure_ascii=False)
            
            # 返回指定分块的内容
            chunk_content = chunks[chunk_index]
            
            return json.dumps({
                "file_path": file_path,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "chunk_content": chunk_content,
                "chunk_lines": chunk_content.count('\n') + 1,
                "is_last_chunk": chunk_index == len(chunks) - 1
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"读取文件分块时发生错误: {str(e)}"}, ensure_ascii=False)
    
    async def _arun(self, file_path: str, chunk_index: int) -> str:
        """异步执行"""
        return await asyncio.to_thread(self._run, file_path, chunk_index)


# 创建工具实例
list_dir_tool = ListDirectoryTool(root_dir=LAW_DOCS_PATH)
get_file_chunks_info_tool = GetFileChunksInfoTool()
read_file_chunk_tool = ReadFileChunkTool()

# 获取所有文件列表工具
class GetAllFilesInput(BaseModel):
    """获取所有文件工具的输入参数"""
    include_size: bool = Field(default=True, description="是否包含文件大小信息")

class GetAllFilesTool(BaseTool):
    """获取所有可用的法律文件路径列表"""
    name: str = "get_all_files"
    description: str = """获取法律文档目录下所有可用的文件路径列表。
    这个工具会递归扫描整个法律文档目录，返回所有文件的相对路径。
    使用此工具可以查看所有可用的文件，避免路径错误。
    
    返回：包含所有文件路径的 JSON 列表，每个文件包含：
    - path: 相对路径（可直接用于 smart_read_file）
    - size: 文件大小（字节）
    - type: 文件类型（扩展名）
    """
    args_schema: type[BaseModel] = GetAllFilesInput
    root_dir: str = LAW_DOCS_PATH
    
    def _run(self, include_size: bool = True) -> str:
        """扫描并返回所有文件路径"""
        try:
            files_info = []
            
            # 递归遍历目录
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    # 获取完整路径
                    full_path = os.path.join(root, file)
                    # 计算相对路径
                    rel_path = os.path.relpath(full_path, self.root_dir)
                    # 统一使用正斜杠
                    rel_path = rel_path.replace('\\', '/')
                    
                    file_info = {
                        "path": rel_path,
                        "type": os.path.splitext(file)[1]
                    }
                    
                    if include_size:
                        file_info["size"] = os.path.getsize(full_path)
                    
                    files_info.append(file_info)
            
            # 按路径排序
            files_info.sort(key=lambda x: x["path"])
            
            return json.dumps({
                "total_files": len(files_info),
                "files": files_info
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"扫描文件时发生错误: {str(e)}"}, ensure_ascii=False)
    
    async def _arun(self, include_size: bool = True) -> str:
        """异步执行"""
        return await asyncio.to_thread(self._run, include_size)

# 创建工具实例
get_all_files_tool = GetAllFilesTool()

# --- Legal Retriever Agent 定义 ---

legal_retriever_prompt = """你是一个专业的法律文件检索智能体。你的核心任务是根据用户的法律问题或关键词，从法律文档库中检索相关的法律条文。

## 可用工具详解

### 1. get_all_files
**功能**：递归扫描法律文档目录，返回所有可用文件的列表。

**使用方法**：
```python
get_all_files(include_size=True)
```

**返回内容**：包含所有文件的路径、大小和类型信息。

**使用场景**：
- 首次检索时列出所有可用法律文件
- 根据关键词筛选目标文件路径
- 确认文件是否存在

---

### 2. get_file_chunks_info
**功能**：获取文件的分块元数据（不读取实际内容）。

**使用方法**：
```python
get_file_chunks_info(file_path="civil-code/part1/chapter3.md")
```

**返回内容**：
- `total_chunks`: 文件被分成多少块
- `total_lines`: 文件总行数
- `file_size_bytes`: 文件大小
- `chunk_summaries`: 每个分块的预览信息（包括块索引、行数、前几行内容预览）

**使用场景**：
- 在读取文件内容前，先了解文件结构
- 查看文件有多少分块，决定读取哪些块
- 通过预览信息判断哪些分块可能包含目标条文

---

### 3. read_file_chunk
**功能**：读取文件的指定分块内容。

**使用方法**：
```python
read_file_chunk(
    file_path="civil-code/part1/chapter3.md",
    chunk_index=0  # 读取第 0 块
)
```

**返回内容**：
- `chunk_content`: 该分块的完整内容
- `chunk_lines`: 该分块的行数
- `chunk_index`: 当前分块索引
- `total_chunks`: 总分块数
- `is_last_chunk`: 是否为最后一块

**使用场景**：
- 根据 `get_file_chunks_info` 返回的预览，选择性读取相关分块
- 逐块检索法律条文

**重要提示**：
- 所有文件路径必须是相对路径（如 `civil-code/chapter1.md`）
- 路径使用正斜杠 `/` 分隔
- 直接使用 `get_all_files` 返回的路径即可

## 工作流程

### 步骤1：理解检索需求
- 分析用户问题，提取关键法律概念
- 例如："机关法人"、"民事活动"、"合同效力"、"违约金"等

### 步骤2：定位相关文件
```python
# 调用 get_all_files 获取所有文件
files = get_all_files(include_size=True)

# 根据关键词筛选相关文件
# 例如：搜索"机关法人" → 选择 civil-code/part1/chapter3.md
```

### 步骤3：获取文件分块信息
```python
# 先获取文件的分块元数据
info = get_file_chunks_info(file_path="civil-code/part1/chapter3.md")

# 返回示例：
# {
#   "total_chunks": 5,
#   "total_lines": 234,
#   "chunk_summaries": [
#     {
#       "chunk_index": 0,
#       "lines": 48,
#       "preview": "第一节 一般规定\n第五十七条 法人是具有民事权利能力..."
#     },
#     ...
#   ]
# }
```

### 步骤4：选择性读取分块
```python
# 根据预览信息，选择可能包含目标条文的分块
chunk = read_file_chunk(
    file_path="civil-code/part1/chapter3.md",
    chunk_index=2  # 例如预览显示第2块可能包含相关内容
)

# 在 chunk_content 中搜索关键词，提取相关法条
```

### 步骤5：提取和返回结果
从分块内容中提取相关法条，返回 JSON 格式：

```json
{
  "retrieval_status": "success",
  "keywords_used": "机关法人 民事活动",
  "files_searched": ["civil-code/part1/chapter3.md"],
  "results": [
    {
      "article_number": "第九十七条",
      "article_content": "有独立经费的机关和承担行政职能的法定机构从成立之日起，具有机关法人资格，可以从事为履行职能所需要的民事活动。",
      "relevance_score": 0.95,
      "relevance_reason": "该条文直接规定了机关法人可以从事为履行职能所需的民事活动"
    }
  ],
  "total_results": 1
}
```

## 检索策略

1. **广度优先**：先用 `get_file_chunks_info` 快速浏览多个文件的预览
2. **精准定位**：根据预览信息，只读取最相关的分块
3. **关键词匹配**：在分块内容中搜索关键词，提取条文
4. **迭代优化**：如果结果不理想，调整关键词或查看其他分块

## 错误处理

- **文件不存在**：重新调用 `get_all_files` 确认路径
- **分块索引越界**：检查 `get_file_chunks_info` 返回的 `total_chunks`
- **路径格式错误**：确保使用相对路径和正斜杠 `/`
"""

legal_retriever_subagent = {
    "name": "legal-retriever-agent",
    "description": "用于从法律文档库中检索相关法律条文。接收关键词或法律问题，返回 Top-K 相关条文的 JSON 列表。",
    "system_prompt": legal_retriever_prompt,
    "tools": [get_all_files_tool, get_file_chunks_info_tool, read_file_chunk_tool],
}

legal_evaluator_prompt="""
你需要执行以下步骤（CoT 推理）：
1. **理解用户问题**：提取用户问题中的核心法律要素（如合同类型、争议焦点、诉求等）
2. **逐条分析候选法条**：
   - 分析每条法律条文的适用范围和法律规定
   - 将法条内容与用户问题进行逻辑比对
   - 判断法条是否直接适用或相关
3. **做出评估决策**：
   - 如果找到明确适用的法条 → 返回 MATCH
   - 如果所有候选法条都不相关或依据不足 → 返回 MISMATCH

返回格式必须是严格的 JSON：
{{
    \"status\": \"MATCH\" 或 \"MISMATCH\",
    \"reason\": \"详细的匹配/不匹配理由，包含 CoT 推理过程\",
    \"citation\": \"确定的法律条文引用（如《民法典》第585条），仅在 MATCH 时填写\",
    \"confidence\": 0.0-1.0 的置信度评分,
    \"thinking_process\": \"逐条分析的思考过程\"
}}

示例 - 匹配情况：
{{
    \"status\": \"MATCH\",
    \"reason\": \"用户问题关于违约金过高需要降低，检索到的第585条明确规定'约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少'，与用户诉求完全吻合。\",
    \"citation\": \"《中华人民共和国民法典》第585条\",
    \"confidence\": 0.95,
    \"thinking_process\": \"第585条讲的是违约金调整，用户依据也是违约金过高要求降低，两者逻辑一致。第584条虽然也涉及违约责任，但主要讲的是违约后的损害赔偿，与违约金调整不直接相关。因此第585条是最佳匹配。\"
}}

示例 - 不匹配情况：
{{
    \"status\": \"MISMATCH\",
    \"reason\": \"检索到的条文主要涉及合同订立和合同效力，但用户问题关注的是合同履行后的违约责任，两者焦点不一致。建议调整检索关键词，聚焦'违约责任'、'损害赔偿'等更精准的术语。\",
    \"citation\": \"\",
    \"confidence\": 0.3,
    \"thinking_process\": \"检索到第469-471条都是关于合同订立的形式要件，用户问的是违约后的救济措施，这些条文无法作为依据。需要重新检索违约责任相关条文。\"
}}

请确保你的评估严谨、逻辑清晰，并提供充分的推理依据。
"""

legal_evaluator_subagent = {
    "name": "legal-evaluator-agent",
    "description": "用于评估检索到的法律条文是否与用户问题匹配。接收用户问题和候选法条列表，返回包含评估结果的 JSON 对象（status、reason、citation、confidence）。",
    "system_prompt": legal_evaluator_prompt,
    "tools": [],  # 纯逻辑推理，不使用外部工具
}

# --- 主智能体定义 ---

# LegalContractCoordinator (法律合同评估协调者)
# 职责：协调检索和评估子智能体，管理循环重试逻辑，最终生成答案
legal_coordinator_instructions = """你是一个专业的法律合同评估协调器。你的核心任务是接收用户的法律合同相关问题，通过协调子智能体找到对应的法律条文编号（第几条）。

你的工作流程如下：

1. **接收用户输入**：理解用户的法律问题或合同条款争议点

2. **生成检索关键词**：
   - 分析用户问题，提取核心法律概念
   - 生成精准的检索关键词（如\"合同 违约金 过高\"）
   -第一次尝试使用用户问题中的直接关键词

3.调用检索子智能体**：
   - 使用 `legal-retriever-agent` 检索相关法律条文
   - 获取 Top-K 候选条文列表

4. **调用评估子智能体**：
   - 使用 `legal-evaluator-agent` 评估检索结果
   - 接收评估报告（MATCH 或 MISMATCH）

5. **循环决策（核心逻辑）**：
   - **如果评估结果是 MATCH**：
     * 汇总信息，生成最终答案
     * 包含：法律条文编号、条文内容、适用理由
     * 结束流程
   
   - **如果评估结果是 MISMATCH**：
     * 分析不匹配的原因（评估子智能体会提供建议）
     * **重写检索关键词**（调整方向、更换术语、扩展/缩小范围）
     * 再次调用检索子智能体（进入下一轮循环）
     * **重试次数限制**：最多 3 次循环

   - **如果 3 次仍无果**：
     * 输出\"无法找到确切法律依据\"
     * 提供可能的原因和建议

6. **最终输出格式**：
   成功时：
   ```
   根据您提供的合同信息，相关法律依据如下：

   **法律条文**：《中华人民共和国民法典》第XXX条

   **条文内容**：[完整条文内容]

   **适用理由**：[为什么这条法律适用于用户的问题]

   **检索过程**：共进行了 X 次检索，最终匹配成功。
   ```

   失败时：
   ```
   经过 3 次深度检索和评估，未能找到与您问题完全匹配的法律条文。

   **可能原因**：
   - [分析原因]

   **建议**：
   - [给出建议，如重新描述问题、提供更多背景信息等]
   ```

**重要提示**：
- 立即开始执行，不要等待用户确认
- 使用 `write_todos` 工具记录检索计划（可选）
- 每次重试必须调整检索策略，不要重复相同的关键词
- 充分利用评估子智能体的反馈优化检索方向
- 所有操作在内存中进行，不需要使用文件系统工具
"""

def create_legal_contract_agent(model: BaseChatModel) -> Runnable:
    """
    创建法律合同评估智能体。
    
    Args:
        model: 用于智能体的语言模型实例。
    
    Returns:
        一个 Agent 实例，能够协调检索和评估子智能体，找到相关法律条文。
    """
    agent = create_deep_agent(
        model=model,
        system_prompt=legal_coordinator_instructions,
        subagents=[legal_retriever_subagent, legal_evaluator_subagent],
        tools=[]  # 协调者不直接使用工具，通过子智能体调用
    )
    return agent


# --- 测试用例 ---

async def handle_legal_contract_evaluation(state):
    """
    处理法律合同评估的异步函数。
    """
    # 初始化语言模型
    llm = ChatOpenAI(
        model="qwen3-coder",
        temperature=0.2,
    )

    # 创建法律合同评估智能体
    legal_agent = create_legal_contract_agent(llm)

    # 用户的法律问题示例
    user_messages = state.get("messages", [])  
    print(user_messages)
    # user_legal_question = """请找到具体的法律条文编号。相关法律内容如下：机关法人可以从事为履行职能所需的民事活动;《最高人民法院关于适用(民法典》合同编通则的解释》第11条:签约主体应具备相应法律资格，"""

    print("\n" + "="*80)
    print("法律合同评估智能体测试")
    print("="*80)
    print("\n【用户问题】")
    print(user_messages)
    print("\n" + "-"*80)
    print("【智能体开始分析】\n")

    try:
        # 调用智能体进行评估
        result = await legal_agent.ainvoke({
            "messages": user_messages
        })
        
        print("\n" + "="*80)
        print(f"【分析完成】: {result}")
        print("="*80)
        return {  
                "messages": state["messages"] + result["messages"],  # Add AI response to messages  
                # Optionally add todos/files if your agent creates them  
            }        
    except Exception as e:
        print(f"\n--- 智能体执行出错 ---")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

# --- LangGraph 状态图配置 ---

class Context(TypedDict):
    context: Dict[str, Any]
    messages: Annotated[list, add_messages]  
    # 如果需要文件,可以添加  
    files: dict[str, str]  # 文件路径 -> 文件内容    


builder = StateGraph(Context)
builder.add_node("legal_evaluation", handle_legal_contract_evaluation)
builder.add_edge("legal_evaluation", END)
builder.set_entry_point("legal_evaluation")
graph = builder.compile()

memory = MemorySaver()
def make_graph(config: RunnableConfig):  
    # 根据配置动态构建图  
    return builder.compile(checkpointer=memory)
async def main():
    """主函数：运行法律合同评估流程"""
    print("\n启动法律合同评估系统...\n")
    
    async for namespace, stream_mode, data in graph.astream(
        input={"context": {}},
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True
    ):
        # 流式输出中间结果
        if stream_mode == "messages":
            print(f"[{namespace}] {data}")

# langgraph dev --host 0.0.0.0 --port 22204 
if __name__ == "__main__":
    asyncio.run(main())
