

import time
import os
from basereal import BaseReal
from logger import logger
from openai import OpenAI

# 处理 LLM 流式响应，并按标点合理分块
def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    end = time.perf_counter()
    logger.info(f"llm Time init: {end - start:.3f}s")

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': 'huanghe assistant'},#
            {'role': 'user', 'content': message}
        ],
        stream=True,
        stream_options={"include_usage": True}
    )

    result = ""
    first = True
    punctuation_end = "。！？"  # 句子终止符，立即发送
    punctuation_middle = "，；："  # 句子中间停顿符，继续累积

    total_tokens = 0  # 记录所有 token 数量

    for chunk in completion:
        if chunk.choices:
            if first:
                logger.info(f"llm Time to first chunk: {time.perf_counter() - start:.3f}s")
                first = False

            msg = chunk.choices[0].delta.content
            total_tokens += len(msg)  # 统计字符数作为 token 数近似值（若有 API 提供 token 计数更好）

            lastpos = 0
            for i, char in enumerate(msg):
                if char in punctuation_end:  
                    result += msg[lastpos:i+1]
                    lastpos = i+1
                    logger.info(f"End Chunk: {result}")
                    nerfreal.put_msg_txt(result)
                    result = ""  
                elif char in punctuation_middle:
                    result += msg[lastpos:i+1]
                    lastpos = i+1

            result += msg[lastpos:]

    if result:
        logger.info(f"Final Chunk: {result}")
        nerfreal.put_msg_txt(result)

    total_time = time.perf_counter() - start  # 总时间
    tps = total_tokens / total_time if total_time > 0 else 0  # 计算 TPS
    logger.info(f"LLM Total Tokens: {total_tokens}, Time: {total_time:.3f}s, TPS: {tps:.2f} tokens/s")
