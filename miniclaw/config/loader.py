from pydantic_settings import BaseSettings
from pathlib import Path

class Config(BaseSettings):
    class Providers(BaseSettings):
        openai: dict = {"apikey": ""}
    
    class Agents(BaseSettings):
        class Defaults(BaseSettings):
            model: str = "gpt-4.1-mini"
            provider: str = "openai"
            base_url: str = "https://api.chatanywhere.tech"
    
        default: Defaults = Defaults() 
    
    providers: Providers = Providers()
    agents: Agents = Agents()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path=None):
    return Config()
