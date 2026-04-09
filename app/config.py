from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "Salesforce/codet5p-220m"
    device: str | None = None  # None이면 자동 감지
    max_token_size: int = 512
    stride: int = 256
    batch_size: int = 16
    use_fp16: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
