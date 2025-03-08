"""
Validate the parsed data using Pandas
"""
import math
import pandas as pd
import datetime as dt
from data_structs import (
    Statement,
    Account,
    Transaction,
    DATE,
    DEPOSITS,
    WITHDRAWLS,
    DAILY_BALANCE,
    RUNNING_BALANCE,
    CHECK_NUMBER,
    DESCRIPTION,
    ACCOUNT_NUMBER,
    ACCOUNT_TYPE,
    DEBUG_STR
)


def dataframe_validate_amounts(
    statement: Statement,
    account: Account,
    monies: pd.DataFrame
) -> None:
    date: dt.date = dt.datetime.strptime(statement.statement_date, "%B %d, %Y").date()
    total_deposits: float = float(account.total_deposits.replace(",", ""))
    total_withdrawls: float = float(account.total_withdrawls.replace(",", ""))
    beginning_balance: float = float(account.beginning_balance.replace(",", ""))
    ending_balance: float = float(account.ending_balance.replace(",", ""))
    beginning_date: dt.date = dt.datetime.strptime(
        account.beginning_date_mmyy + f"/{date.year}",
        "%m/%d/%Y"
    ).date()
    ending_date: dt.date = dt.datetime.strptime(
        account.ending_date_mmyy + f"/{date.year}",
        "%m/%d/%Y"
    ).date()

    assert date == ending_date

    running_balance: float = beginning_balance

    for row in monies.itertuples(index=True):
        running_balance += row.deposits - row.withdrawls
        if not math.isnan(row.daily_balance):
            assert round(running_balance, 2) == round(row.daily_balance, 2)

    assert round(running_balance, 2) == round(ending_balance, 2)

    sum_deposits: float = monies[DEPOSITS].sum()
    sum_withdrawls: float = monies[WITHDRAWLS].sum()

    assert round(total_deposits, 2) == round(sum_deposits, 2)
    assert round(total_withdrawls, 2) == round(sum_withdrawls, 2)


def validate_transaction(t: Transaction) -> None:
    assert t.date
    assert t.month
    assert t.day
    assert t.description
    assert t.deposits or t.withdrawls
    assert t.debug_str


def validate_account(a: Account) -> None:
    assert a.account_type
    assert a.account_num
    assert a.account_owner
    assert a.beginning_balance
    assert a.ending_balance
    assert a.total_deposits
    assert a.total_withdrawls
    assert a.beginning_date
    assert a.beginning_date_mmyy
    assert a.ending_date
    assert a.ending_date_mmyy
    assert a.df_transactions is not None

    assert a.transaction_history
    for trans in a.transaction_history:
        validate_transaction(trans)


def validate_statement(s: Statement) -> None:
    assert s.statement_date
    assert s.file_path
    assert s.file_name
    assert s.statement_date
    assert s.start_date
    assert s.end_date
    assert s.fee_period_dates
    assert s.df_transactions_combined is not None

    assert s.accounts
    for account in s.accounts:
        validate_account(account)
        dataframe_validate_amounts(s, account, account.df_transactions)
