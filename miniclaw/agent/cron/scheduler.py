"""Cron scheduler with timezone support."""
import asyncio
from datetime import datetime
from typing import Dict, Any, Callable
from croniter import croniter
import pytz
from pathlib import Path

from .models import CronJob, JobConfig, JobConfigManager

class CronScheduler:
    """定时任务调度器"""
    
    def __init__(self, config_path: str = None,bus=None,memory_store=None):
        if config_path is None:
            config_path = Path.home() / ".miniclaw" / "cron" / "jobs.json"
        self._config_path = Path(config_path).expanduser()
         # 可选：确保目录存在
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_manager = JobConfigManager(self._config_path)
        self._tasks: Dict[str, asyncio.Task] = {}  # {job_id: task}
        self._executor = TaskExecutor(bus,memory_store)
        self._bus = bus
        self._memory_store = memory_store

    async def start(self):
        """启动调度器"""
        config = self._config_manager.load()
        for job in config.jobs:
            if job.enabled:
                await self._start_job(job)
        
        # 启动配置文件监听（热加载）
        asyncio.create_task(self._watch_config())

    async def _start_job(self, job: CronJob):
        """启动单个任务"""
        if job.id in self._tasks:
            self._tasks[job.id].cancel()
        
        task = asyncio.create_task(self._run_job(job))
        self._tasks[job.id] = task
        print(f"⏰ 已启动任务: {job.id} ({job.schedule.expr})")

    async def _run_job(self, job: CronJob):
        """执行任务循环"""
        tz = pytz.timezone(job.schedule.tz)
        
        while True:
            try:
                # 验证 cron 表达式
                if not self._validate_cron_expr(job.schedule.expr):
                    print(f"❌ 任务 {job.id} 的 cron 表达式无效: {job.schedule.expr}")
                    print(f"   支持标准 Unix cron 格式，如 '0 12 * * *' (每天12点)")
                    return  # 停止这个任务
                
                # 计算下次执行时间
                now = datetime.now(tz)
                cron = croniter(job.schedule.expr, now)
                next_run = cron.get_next(datetime)
                
                # 添加随机延迟（可选）
                delay = (next_run - datetime.now(tz)).total_seconds()
                if job.schedule.jitter > 0:
                    delay += asyncio.get_event_loop().random() * job.schedule.jitter
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # 执行任务
                await self._execute_with_retry(job)
                
            except Exception as e:
                print(f"❌ 任务 {job.id} 执行异常: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再重试
    
    def _validate_cron_expr(self, expr: str) -> bool:
        """验证 cron 表达式是否有效"""
        # 不支持 Quartz 格式（带 ? 或 L 等）
        if '?' in expr or 'L' in expr.upper():
            return False
        try:
            croniter(expr)
            return True
        except Exception:
            return False

    async def _execute_with_retry(self, job: CronJob):
        """带重试的任务执行"""
        for attempt in range(job.retry.max_attempts):
            try:
                await self._executor.execute(job.payload)
                print(f"✅ 任务 {job.id} 执行成功")
                return
            except Exception as e:
                print(f"⚠️ 任务 {job.id} 第 {attempt+1} 次尝试失败: {e}")
                if attempt < job.retry.max_attempts - 1:
                    delay = job.retry.delay_seconds * (job.retry.backoff_factor ** attempt)
                    await asyncio.sleep(delay)
        
        print(f"❌ 任务 {job.id} 重试 {job.retry.max_attempts} 次后失败")

    async def _watch_config(self):
        """监听配置文件变化（简化实现）"""
        last_mtime = 0
        while True:
            await asyncio.sleep(30)
            try:
                mtime = self._config_path.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    await self._reload_config()
            except Exception:
                pass
    
    async def _reload_config(self):
        """热重载配置"""
        print("🔄 检测到配置变化，重新加载...")
        config = self._config_manager.load()
        
        # 停止已删除或禁用的任务
        for job_id in list(self._tasks.keys()):
            job = next((j for j in config.jobs if j.id == job_id), None)
            if not job or not job.enabled:
                self._tasks[job_id].cancel()
                del self._tasks[job_id]
                print(f"⏹️ 已停止任务: {job_id}")
        
        # 启动新增或修改的任务
        for job in config.jobs:
            if job.enabled and job.id not in self._tasks:
                await self._start_job(job)

    def add_job(self, job: CronJob):
        """添加新任务"""
        config = self._config_manager.load()
        config.jobs.append(job)
        self._config_manager.save(config)

    def remove_job(self, job_id: str):
        """删除任务"""
        config = self._config_manager.load()
        config.jobs = [j for j in config.jobs if j.id != job_id]
        self._config_manager.save(config)

    def get_jobs(self) -> list[CronJob]:
        """获取所有任务"""
        return self._config_manager.load().jobs


class TaskExecutor:
    """任务执行器"""
    def __init__(self, bus=None,memory_store=None):
        self._bus = bus
        self._memory_store = memory_store

    async def execute(self, payload):
        """执行任务（支持 JobPayload 对象或 dict）"""
        # 统一转换为 dict
        if hasattr(payload, '__dict__'):
            payload_dict = {k: v for k, v in payload.__dict__.items() if not k.startswith('_')}
        else:
            payload_dict = payload
        
        kind = payload_dict.get("kind", "command")
        
        if kind == "command":
            await self._execute_command(payload_dict)
        elif kind == "webhook":
            await self._execute_webhook(payload_dict)
        elif kind == "internal":
            await self._execute_internal(payload_dict)
        else:
            raise ValueError(f"未知任务类型: {kind}")

    async def _execute_command(self, payload: dict):
        """执行系统命令"""
        import subprocess
        command = payload.get("message", "")
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"命令执行失败: {result.stderr}")

    async def _execute_webhook(self, payload: dict):
        """发送 HTTP 请求"""
        import httpx
        url = payload.get("url")
        method = payload.get("method", "POST")
        headers = payload.get("headers", {})
        message = payload.get("message", "")
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json={"message": message} if message else {}
            )
            response.raise_for_status()

    async def _execute_internal(self, payload: dict):
        """执行内部操作"""
        message = payload.get("message", "")
        # 这里可以调用内部服务，如通知用户、更新记忆等
        if self._memory_store:
            self._memory_store.append_history(
                f"[Cron Task] {message}",
                session_key="system",
                kind="cron"
            )
       