from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime

class User(SQLModel, table=True):
    user_id: str = Field(primary_key=True, max_length=36)
    username: str = Field(unique=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now, nullable=False)
    last_login: Optional[datetime] = Field(default=None)
    transactions: List["Transaction"] = Relationship(back_populates="user")

class Transaction(SQLModel, table=True):
    transaction_id: str = Field(primary_key=True, max_length=36)
    user_id: Optional[str] = Field(default=None, foreign_key="user.user_id", max_length=36)
    transaction_datetime: datetime = Field(default_factory=datetime.now, nullable=False)
    stock_symbol: str = Field(default="", max_length=20)
    transaction_type: str = Field(default="", max_length=10)
    quantity: int = Field(default=0)
    price: float = Field(default=0.0)
    commission_local: float = Field(default=0.0)
    processed: bool = Field(default=True)
    user: Optional[User] = Relationship(back_populates="transactions")
