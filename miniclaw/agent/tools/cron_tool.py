"""Cron job management tools."""
from typing import Any
from pathlib import Path

from miniclaw.agent.tools.base import Tool, tool_parameters
from miniclaw.agent.tools.schema import StringSchema, IntegerSchema, tool_parameters_schema
from miniclaw.agent.cron.models import CronJob, CronSchedule, JobPayload, RetryConfig

@tool_parameters(
    tool_parameters_schema(
        action=StringSchema("操作类型", enum=["add", "remove", "list", "enable", "disable"]),
        job_id=StringSchema("任务ID", nullable=True),
        name=StringSchema("任务名称", nullable=True),
        expr=StringSchema("Cron表达式", nullable=True),
        tz=StringSchema("时区", nullable=True),
        kind=StringSchema("任务类型", enum=["command", "webhook", "internal"], nullable=True),
        message=StringSchema("消息/命令内容", nullable=True),
        url=StringSchema("Webhook URL", nullable=True),
        required=["action"],
    )
)
class CronTool(Tool):
    """定时任务管理工具"""
    
    def __init__(self, scheduler):
        self._scheduler = scheduler
    
    @property
    def name(self) -> str:
        return "cron_job"
    
    @property
    def description(self) -> str:
        return "管理定时任务：添加、删除、查询、启用、禁用"
    
    async def execute(self, action: str, **kwargs: Any) -> str:
        """执行任务管理操作"""
        if action == "add":
            return await self._add_job(kwargs)
        elif action == "remove":
            return self._remove_job(kwargs.get("job_id"))
        elif action == "list":
            return self._list_jobs()
        elif action == "enable":
            return await self._toggle_job(kwargs.get("job_id"), True)
        elif action == "disable":
            return await self._toggle_job(kwargs.get("job_id"), False)
        else:
            return f"未知操作: {action}"
    
    async def _add_job(self, params: dict) -> str:
        """添加新任务"""
        job_id = params.get("job_id") or params.get("name", "unnamed")
        expr = params.get("expr", "0 12 * * *")
        tz = params.get("tz", "Asia/Shanghai")
        kind = params.get("kind", "internal")
        message = params.get("message", "")
        url = params.get("url")
        
        job = CronJob(
            id=job_id,
            name=params.get("name", job_id),
            enabled=True,
            schedule=CronSchedule(expr=expr, tz=tz),
            payload=JobPayload(kind=kind, message=message, url=url),
        )
        
        self._scheduler.add_job(job)
        
        # 立即启动任务
        await self._scheduler._start_job(job)
        
        return f"✅ 已添加定时任务\nID: {job_id}\n表达式: {expr}\n时区: {tz}"
    
    def _remove_job(self, job_id: str) -> str:
        """删除任务"""
        if not job_id:
            return "❌ 需要提供 job_id"
        
        self._scheduler.remove_job(job_id)
        return f"✅ 已删除任务: {job_id}"
    
    def _list_jobs(self) -> str:
        """列出所有任务"""
        jobs = self._scheduler.get_jobs()
        if not jobs:
            return "暂无定时任务"
        
        lines = ["📋 定时任务列表:"]
        for job in jobs:
            status = "🟢" if job.enabled else "🔴"
            lines.append(f"{status} {job.id} - {job.name}")
            lines.append(f"   表达式: {job.schedule.expr}")
            lines.append(f"   时区: {job.schedule.tz}")
            lines.append(f"   类型: {job.payload.kind}")
        
        return "\n".join(lines)
    
    async def _toggle_job(self, job_id: str, enabled: bool) -> str:
        """启用/禁用任务"""
        if not job_id:
            return "❌ 需要提供 job_id"
        
        config = self._scheduler._config_manager.load()
        job = next((j for j in config.jobs if j.id == job_id), None)
        
        if not job:
            return f"❌ 任务 '{job_id}' 不存在"
        
        job.enabled = enabled
        self._scheduler._config_manager.save(config)
        
        if enabled:
            await self._scheduler._start_job(job)
            return f"✅ 已启用任务: {job_id}"
        else:
            if job_id in self._scheduler._tasks:
                self._scheduler._tasks[job_id].cancel()
                del self._scheduler._tasks[job_id]
            return f"✅ 已禁用任务: {job_id}"
