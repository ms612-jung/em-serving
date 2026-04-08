from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "Salesforce/codet5p-220m"
    device: str | None = None  # None이면 자동 감지
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
