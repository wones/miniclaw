"""Cron job configuration models."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import json

@dataclass
class CronSchedule:
    """定时调度配置"""
    kind: str = "cron"
    expr: str = ""  # cron 表达式
    tz: str = "Asia/Shanghai"  # 时区
    jitter: int = 0  # 随机延迟秒数（避免并发峰值）

@dataclass
class JobPayload:
    """任务执行内容"""
    kind: str = "command"  # command, webhook, internal
    message: str = ""  # 命令/消息内容
    url: Optional[str] = None  # webhook URL
    headers: Dict[str, str] = field(default_factory=dict)  # 请求头
    method: str = "POST"  # HTTP 方法

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    delay_seconds: int = 60
    backoff_factor: float = 2.0

@dataclass
class CronJob:
    """定时任务定义"""
    id: str
    name: str = ""
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=CronSchedule)
    payload: JobPayload = field(default_factory=JobPayload)
    retry: RetryConfig = field(default_factory=RetryConfig)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class JobConfig:
    """任务配置文件结构"""
    version: str = "1.0"
    jobs: list[CronJob] = field(default_factory=list)

class JobConfigManager:
    """任务配置管理器"""
    
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> JobConfig:
        """加载配置文件"""
        if not self._config_path.exists():
            return JobConfig()
        
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._from_dict(data)
        except Exception:
            return JobConfig()
    
    def save(self, config: JobConfig):
        """保存配置文件"""
        config.updated_at = datetime.now()
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(config), f, ensure_ascii=False, indent=2)
    
    def _from_dict(self, data: dict) -> JobConfig:
        """从字典解析配置"""
        jobs = []
        for job_data in data.get("jobs", []):
            job = CronJob(
                id=job_data["id"],
                name=job_data.get("name", ""),
                enabled=job_data.get("enabled", True),
                schedule=CronSchedule(**job_data.get("schedule", {})),
                payload=JobPayload(**job_data.get("payload", {})),
                retry=RetryConfig(**job_data.get("retry", {})),
            )
            jobs.append(job)
        return JobConfig(
            version=data.get("version", "1.0"),
            jobs=jobs
        )
    
    def _to_dict(self, config: JobConfig) -> dict:
        """转换为字典"""
        return {
            "version": config.version,
            "updated_at": datetime.now().isoformat(),
            "jobs": [{
                "id": job.id,
                "name": job.name,
                "enabled": job.enabled,
                "schedule": {
                    "kind": job.schedule.kind,
                    "expr": job.schedule.expr,
                    "tz": job.schedule.tz,
                    "jitter": job.schedule.jitter,
                },
                "payload": {
                    "kind": job.payload.kind,
                    "message": job.payload.message,
                    "url": job.payload.url,
                    "headers": job.payload.headers,
                    "method": job.payload.method,
                },
                "retry": {
                    "max_attempts": job.retry.max_attempts,
                    "delay_seconds": job.retry.delay_seconds,
                    "backoff_factor": job.retry.backoff_factor,
                },
            } for job in config.jobs]
        }
