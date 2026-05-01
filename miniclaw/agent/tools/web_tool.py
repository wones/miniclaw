"""Network tools for HTTP requests, web search, and web page fetching.

Based on miniclaw's design with async HTTP client support.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from bs4 import BeautifulSoup

from miniclaw.agent.tools.base import Tool, tool_parameters
from miniclaw.agent.tools.schema import StringSchema, IntegerSchema, tool_parameters_schema


# ---------------------------------------------------------------------------
# HTTP Request Tool
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        url=StringSchema("目标URL地址", min_length=10),
        method=StringSchema(
            "HTTP请求方法",
            enum=["GET", "POST", "PUT", "DELETE"],
        ),
        headers=StringSchema("请求头，JSON格式字符串，如{\"Content-Type\":\"application/json\"}", nullable=True),
        body=StringSchema("请求体，JSON格式字符串，仅POST/PUT请求使用", nullable=True),
        timeout=IntegerSchema(30, description="超时时间（秒）", minimum=1, maximum=120),
        required=["url"],
    )
)
class HttpRequestTool(Tool):
    """发送HTTP请求获取远程API数据"""

    _MAX_RESPONSE_CHARS = 50_000
    _USER_AGENT = "miniclaw-agent/1.0"

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return "发送HTTP请求获取远程API数据，支持GET/POST/PUT/DELETE方法"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, url: str, method: str = "GET", headers: str | None = None,
                     body: str | None = None, timeout: int = 30, **kwargs: Any) -> str:
        """执行HTTP请求"""
        try:
            headers_dict = {}
            if headers:
                try:
                    headers_dict = json.loads(headers)
                except json.JSONDecodeError:
                    return "Error: 请求头格式无效，应为JSON字符串"

            if "User-Agent" not in headers_dict:
                headers_dict["User-Agent"] = self._USER_AGENT

            body_data = None
            if body and method in ("POST", "PUT"):
                try:
                    body_data = json.loads(body)
                except json.JSONDecodeError:
                    return "Error: 请求体格式无效，应为JSON字符串"

            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=headers_dict,
                    json=body_data,
                )

                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        content = json.dumps(response.json(), ensure_ascii=False, indent=2)
                    except Exception:
                        content = response.text
                else:
                    content = response.text

                if len(content) > self._MAX_RESPONSE_CHARS:
                    content = content[:self._MAX_RESPONSE_CHARS // 2] + \
                              f"\n\n[... 内容过长，已截断 {len(content) - self._MAX_RESPONSE_CHARS} 字符 ...]\n\n" + \
                              content[-self._MAX_RESPONSE_CHARS // 2:]

                return f"HTTP {response.status_code}\nContent-Type: {content_type}\n\n{content}"

        except httpx.HTTPError as e:
            return f"Error: HTTP请求失败 - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("搜索关键词", min_length=1, max_length=500),
        count=IntegerSchema(5, description="返回结果数量", minimum=1, maximum=20),
        required=["query"],
    )
)
class WebSearchTool(Tool):
    """使用Serper API进行网络搜索"""

    _SERPER_API_URL = "https://google.serper.dev/search"
    _MAX_RESULT_LENGTH = 200

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "通过搜索引擎搜索互联网信息，获取最新的新闻、知识和数据"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, count: int = 5, **kwargs: Any) -> str:
        """执行网络搜索"""
        import os
        
        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            return "Error: 未配置SERPER_API_KEY环境变量，无法进行搜索。请访问 https://serper.dev/ 获取API密钥。"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._SERPER_API_URL,
                    headers={
                        "X-API-KEY": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": count},
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()

                output_lines = [f"搜索结果: {query}"]
                output_lines.append("=" * 50)

                organic = results.get("organic", [])
                for i, result in enumerate(organic[:count], 1):
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    
                    if len(snippet) > self._MAX_RESULT_LENGTH:
                        snippet = snippet[:self._MAX_RESULT_LENGTH] + "..."
                    
                    output_lines.append(f"\n{i}. {title}")
                    output_lines.append(f"   URL: {link}")
                    output_lines.append(f"   摘要: {snippet}")

                news = results.get("news", [])
                if news:
                    output_lines.append("\n📰 相关新闻:")
                    for i, result in enumerate(news[:3], 1):
                        title = result.get("title", "")
                        link = result.get("link", "")
                        source = result.get("source", "")
                        output_lines.append(f"\n   {i}. {title}")
                        output_lines.append(f"      来源: {source}")
                        output_lines.append(f"      URL: {link}")

                if not organic and not news:
                    output_lines.append("\n未找到相关结果")

                return "\n".join(output_lines)

        except httpx.HTTPError as e:
            return f"Error: 搜索失败 - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Web Fetch Tool
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        url=StringSchema("目标网页URL地址", min_length=10),
        timeout=IntegerSchema("超时时间（秒）", minimum=1, maximum=120),
        extract_text_only=StringSchema("是否只提取纯文本", enum=["true", "false"]),
        required=["url"],
    )
)
class WebFetchTool(Tool):
    """获取网页内容"""

    _MAX_CONTENT_CHARS = 100_000
    _USER_AGENT = "miniclaw-agent/1.0"

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "获取网页内容，可选择提取纯文本或保留HTML结构"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, url: str, timeout: int = 30, extract_text_only: str = "true",
                     **kwargs: Any) -> str:
        """获取网页内容"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self._USER_AGENT},
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                
                if extract_text_only.lower() == "true":
                    if "text/html" in content_type:
                        soup = BeautifulSoup(response.text, "html.parser")
                        
                        for script in soup(["script", "style", "nav", "header", "footer"]):
                            script.decompose()
                        
                        text = soup.get_text()
                        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                        content = text
                    else:
                        content = response.text
                else:
                    content = response.text

                if len(content) > self._MAX_CONTENT_CHARS:
                    content = content[:self._MAX_CONTENT_CHARS // 2] + \
                              f"\n\n[... 内容过长，已截断 {len(content) - self._MAX_CONTENT_CHARS} 字符 ...]\n\n" + \
                              content[-self._MAX_CONTENT_CHARS // 2:]

                return f"HTTP {response.status_code}\nContent-Type: {content_type}\n\n{content}"

        except httpx.HTTPError as e:
            return f"Error: 获取网页失败 - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"