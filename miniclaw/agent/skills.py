"""Skills loader for agent capabilities."""
import json
import os
import re
import shutil
from pathlib import Path
import asyncio

# Default builtin skills directory
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills" / "builtin"

# Frontmatter regex
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)

def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

class SkillsLoader:
    """Loader for agent skills."""

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None, disabled_skills: set[str] | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills" / "workspace"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()
        self._skill_cache: dict[str, tuple[float, str, dict]] = {}  # {skill_name: (mtime, content, metadata)}

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """List all available skills."""
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        workspace_names = {entry["name"] for entry in skills}
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=workspace_names)
            )

        if self.disabled_skills:
            skills = [s for s in skills if s["name"] not in self.disabled_skills]

        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def load_skill(self, name: str) -> str | None:
        """Load a skill by name."""
        roots = [self.workspace_skills]
        if self.builtin_skills:
            roots.append(self.builtin_skills)
        for root in roots:
            path = root / name / "SKILL.md"
            if path.exists():
                mtime = path.stat().st_mtime
                # 检查缓存
                if name in self._skill_cache:
                    cached_mtime, cached_content, _ = self._skill_cache[name]
                    if cached_mtime >= mtime:
                        return cached_content
                # 读取文件并缓存
                content = path.read_text(encoding="utf-8")
                metadata = self.get_skill_metadata(name)
                self._skill_cache[name] = (mtime, content, metadata)
                return content
        return None

    async def load_skills_parallel(self, skill_names: list[str]) -> dict[str, str]:
        """并行加载多个技能"""
        async def load_skill_async(name):
            return name, self.load_skill(name)

        tasks = [load_skill_async(name) for name in skill_names]
        results = await asyncio.gather(*tasks)
        return {name: content for name, content in results if content}

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """Load specific skills for inclusion in agent context."""
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        """Build a summary of all skills."""
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        lines: list[str] = ["<skills>"]
        for entry in all_skills:
            skill_name = entry["name"]
            meta = self._get_skill_meta(skill_name)
            available = self._check_requirements(meta)
            lines.extend([
                f'  <skill available="{str(available).lower()}">',
                f"    <name>{_escape_xml(skill_name)}</name>",
                f"    <description>{_escape_xml(self._get_skill_description(skill_name))}</description>",
                f"    <location>{entry['path']}</location>",
            ])
            if not available:
                missing = self._get_missing_requirements(meta)
                if missing:
                    lines.append(f"    <requires>{_escape_xml(missing)}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """Get a description of missing requirements."""
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content

    def _parse_metadata(self, raw: str) -> dict:
        """Parse skill metadata JSON from frontmatter."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        payload = data.get("miniclaw", data.get("openclaw", {}))
        return payload if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """Check if skill requirements are met."""
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict:
        """Get nanobot metadata for a skill."""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_metadata(meta.get("metadata", ""))

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_metadata(meta.get("metadata", "")).get("always")
                or meta.get("always")
            )
        ]

    def get_skill_metadata(self, name: str) -> dict | None:
        """Get metadata from a skill's frontmatter."""
        content = self.load_skill(name)
        if not content or not content.startswith("---"):
            return None
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return None
        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"\'')
        return metadata

    def watch_skills(self, interval: int = 30):
        """启动技能热加载监控"""
        async def watcher():
            while True:
                await asyncio.sleep(interval)
                self._refresh_skill_cache()
    
        asyncio.create_task(watcher())

    def _refresh_skill_cache(self):
        """刷新技能缓存"""      
        all_skills = self.list_skills(filter_unavailable=False)
        for skill in all_skills:        
            name = skill["name"]
            # 检查技能文件是否修改
            roots = [self.workspace_skills]
            if self.builtin_skills:
                roots.append(self.builtin_skills)
        
            for root in roots:
                path = root / name / "SKILL.md"
                if path.exists():
                    mtime = path.stat().st_mtime
                    if name in self._skill_cache:
                        cached_mtime, _, _ = self._skill_cache[name]
                        if mtime > cached_mtime:
                            # 文件已修改，更新缓存
                            content = path.read_text(encoding="utf-8")
                            metadata = self.get_skill_metadata(name)
                            self._skill_cache[name] = (mtime, content, metadata)
                            print(f"Skill '{name}' updated")

    def check_skill_dependencies(self, name: str) -> tuple[bool, list[str]]:
        """检查技能依赖"""
        meta = self._get_skill_meta(name)
        requires = meta.get("requires", {})
    
        missing = []
    
        # 检查 CLI 工具
        for cmd in requires.get("bins", []):
            if not shutil.which(cmd):
                missing.append(f"CLI tool: {cmd}")
    
        # 检查环境变量
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"Environment variable: {env}")
    
        # 检查 Python 包
        for pkg in requires.get("python", []):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(f"Python package: {pkg}")
    
        # 检查其他技能
        for skill in requires.get("skills", []):
            if not self.load_skill(skill):
                missing.append(f"Skill: {skill}")
    
        return len(missing) == 0, missing


    def get_skill_summary(self, name: str) -> str | None:
        """获取技能摘要"""
        content = self.load_skill(name)
        if not content:
            return None
    
        # 提取技能摘要
        lines = content.splitlines()
        summary_lines = []
        in_summary = False
    
        for line in lines:
            if line.startswith("# "):
                if in_summary:
                    break
                else:
                    in_summary = True
            if in_summary:
                summary_lines.append(line)
    
        return "\n".join(summary_lines)

    def load_skills_for_context(self, skill_names: list[str], use_summary: bool = False) -> str:
        """加载技能内容，支持使用摘要"""
        parts = []
        for name in skill_names:
            if use_summary:
                content = self.get_skill_summary(name)
            else:
                content = self.load_skill(name)
        
        if content:
            parts.append(f"### Skill: {name}\n\n{self._strip_frontmatter(content)}")
    
        return "\n\n---\n\n".join(parts)

    def load_skill_safely(self, name: str) -> tuple[str | None, str | None]:
        """安全加载技能，返回内容和错误信息"""
        try:
            content = self.load_skill(name)
            if not content:
                return None, f"Skill '{name}' not found"
        
            # 检查依赖
            is_available, missing = self.check_skill_dependencies(name)
            if not is_available:
                return content, f"Skill '{name}' missing dependencies: {', '.join(missing)}"
        
            return content, None
        except Exception as e:
            return None, f"Error loading skill '{name}': {str(e)}"

    def get_skill_priority(self, name: str) -> int:
        """获取技能优先级"""
        meta = self._get_skill_meta(name)
        return meta.get("priority", 0)

    def get_skills_by_priority(self) -> list[dict[str, str]]:
        """按优先级排序技能"""
        skills = self.list_skills()
        return sorted(skills, key=lambda x: self.get_skill_priority(x["name"]), reverse=True)
