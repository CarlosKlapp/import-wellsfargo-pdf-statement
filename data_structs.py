"""
Data structures for holding the parsed data.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, List, Final
import pandas as pd
import json

ACCOUNT_TYPE: Final[str] = 'acct_name'
ACCOUNT_NUMBER: Final[str] = 'acct_num'
DATE: Final[str] = 'date'
DEPOSITS: Final[str] = 'deposits'
WITHDRAWLS: Final[str] = 'withdrawls'
DAILY_BALANCE: Final[str] = 'daily_balance'
RUNNING_BALANCE: Final[str] = 'running_balance'
CHECK_NUMBER: Final[str] = 'check_number'
DESCRIPTION: Final[str] = 'description'
DEBUG_STR: Final[str] = 'debug_str'
FILE_NAME: Final[str] = 'file_name'
FILE_PATH: Final[str] = 'file_path'
START_DATE: Final[str] = 'start_date'
END_DATE: Final[str] = 'end_date'
FEE_PERIOD_DATES: Final[str] = 'period_dates'
STATEMENT_DATE: Final[str] = 'statement_date'


@dataclass
class Transaction:
    """
    Daily transactions
    """
    date: str | None = None
    month: int | None = None
    day: int | None = None
    check_number: str | None = None
    description: str | None = None
    deposits: str | None = None
    withdrawls: str | None = None
    daily_balance: str | None = None
    debug_str: str | None = None


@dataclass
class Account:
    """
    A single account
    """
    account_type: str | None = None
    account_num: str | None = None
    account_owner: str | None = None
    beginning_balance: str | None = None
    ending_balance: str | None = None
    total_deposits: str | None = None
    total_withdrawls: str | None = None
    beginning_date: str | None = None
    beginning_date_mmyy: str | None = None
    ending_date: str | None = None
    ending_date_mmyy: str | None = None
    transaction_history: List[Transaction] = field(default_factory=list)
    df_transactions: pd.DataFrame | None = None

    def to_json(self):
        # Convert to dictionary and remove excluded fields
        data = asdict(self)
        del data["df_transactions"]  # Remove from root
        return json.dumps(data, cls=DataclassEncoder, indent=4)


@dataclass
class Statement:
    """
    Represents a single PDF file
    """
    file_path: str
    file_name: str
    statement_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    fee_period_dates: str | None = None
    accounts: List[Account] = field(default_factory=list)
    df_transactions_combined: pd.DataFrame | None = None

    def to_json(self):
        # Convert to dictionary and remove excluded fields
        data = asdict(self)
        del data["df_transactions_combined"]  # Remove from root
        return json.dumps(data, cls=DataclassEncoder, indent=4)


class DataclassEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if hasattr(obj, "__dataclass_fields__"):  # Check if it's a dataclass
            data = asdict(obj)
            # Exclude DataFrames
            data = {k: v for k, v in data.items() if not isinstance(v, pd.DataFrame)}
            return data
        return super().default(obj)
