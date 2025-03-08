"""
Main code to extract the data
"""
# from collections.abc import Iterable
import argparse
from dataclasses import dataclass
import inspect
import os
from typing import Any, Final, List, NamedTuple, Tuple, cast, Dict
# import sys
import re
import copy
import datetime as dt
# from dateutil.relativedelta import relativedelta
import pdfplumber
# import pdfplumber.page
import pdfplumber.page
import pymupdf
import pandas as pd
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
    FILE_NAME,
    FILE_PATH,
    START_DATE,
    END_DATE,
    FEE_PERIOD_DATES,
    STATEMENT_DATE,
    DEBUG_STR
)
from validate import validate_statement

# global args
args: argparse.Namespace | None = None

MONTH_NAMES: Final[List[str]] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Names of the months as a tuple, makes search faster.
MONTH_NAMES_AS_TUPLES: Final[Tuple[str, ...]] = tuple(MONTH_NAMES)

# All transaction lines start with a date DD/MM
# Use the beginning date to determine which lines to combine
re_start_transaction_date = re.compile(r"^\d{1,2}\/\d{1,2}$")
# Long date: June 26, 2024
re_match_long_date = re.compile(r"^(?P<long_date>(" + '|'.join(MONTH_NAMES) + r")\s\d{1,2},\s\d{4})")
# Older versions of the statements have the dates in two parts
# Two long date: May 25, 2024 - June 26, 2024
re_match_two_long_date = re.compile(r"(?P<start_long_date>(" + '|'.join(MONTH_NAMES) + r")\s\d{1,2},\s\d{4})\s+-\s+(?P<end_long_date>(" + '|'.join(MONTH_NAMES) + r")\s\d{1,2},\s\d{4})")
# Fee period
re_fee_period = re.compile(r"^(?P<fee_period>Fee period (?P<start>\d{2}/\d{2}/\d{4}) - (?P<end>\d{2}/\d{2}/\d{4}))")


PDF_SECTION_STATEMENT_SUMMARY: Final[str] = "Summary"
PDF_SECTION_TRANSACTIONS: Final[str] = "Transactions"

PDF_SECTION_STATEMENT_SUMMARY_TOKENS: Final[List[str]] = [
    "Statement period activity summary",
    "Activity summary"
]
PDF_SECTION_TRANSACTIONS_TOKENS: Final[List[str]] = [
    "Transaction history"
]


@dataclass
class PdfSectionPosition:
    """
    Simple struct to store the position of each section
    in a PDF page. The algorithm depends on parsing the page
    in the order that each section appears.
    """
    pdf_section: str
    pos: int


def determine_processing_order(
    text: str
) -> List[PdfSectionPosition]:
    """
    Find the different sections inside of the page.

    Args:
        text (str): The PDf text as a single line.

    Returns:
        List[PdfSectionPosition]: Sorted list of section positions.
    """
    pdf_sections: List[PdfSectionPosition] = []

    for section in PDF_SECTION_STATEMENT_SUMMARY_TOKENS:
        # Loop through the text and find all instances
        pos: int = text.find(section)  # -1 if not found
        while pos >= 0:
            # Found the text
            pdf_sections.append(PdfSectionPosition(PDF_SECTION_STATEMENT_SUMMARY, pos))
            # Try to find the next occurrence
            pos = text.find(section, pos + len(section))  # Move the position forward

    for section in PDF_SECTION_TRANSACTIONS_TOKENS:
        # Loop through the text and find all instances
        pos: int = text.find(section)  # -1 if not found
        while pos >= 0:
            # Found the text
            pdf_sections.append(PdfSectionPosition(PDF_SECTION_TRANSACTIONS, pos))
            # Try to find the next occurrence
            pos = text.find(section, pos + len(section))  # Move the position forward

    pdf_sections.sort(key=lambda x: x.pos)

    return pdf_sections


def extract_date(
    lines: List[str]
) -> str:
    """
    Extract the date the statement was generated.
        June 26, 2024 Page 10 of 11

    Args:
        lines(List[str]): PDF as lines of text

    Raises:
        Exception: If not found, throw and exception

    Returns:
        str: date string. Eg. June 26, 2024
    """
    MATCH_NAME: Final[str] = "long_date"
    MATCH_NAME_2: Final[str] = "end_long_date"

    for line in lines:
        if not line:
            # skip blank lines
            continue
        match = re_match_two_long_date.search(line)
        if match:
            return match.group(MATCH_NAME_2)
        if line.startswith(MONTH_NAMES_AS_TUPLES):
            match = re_match_long_date.search(line)
            if match:
                return match.group(MATCH_NAME)
    raise Exception("Unable to find the document date")


class FeePeriodInfo(NamedTuple):
    """
    Fee period info
    """
    full: str
    start: str
    end: str


def extract_fee_period(
    lines: List[str]
) -> FeePeriodInfo | None:
    """
    Extract the fee period.
        Fee period 05/24/2024 - 06/26/2024

    Args:
        lines (List[str]): PDF as lines of text

    Raises:
        Exception: If not found, throw and exception

    Returns:
        FeePeriodInfo: Fee info
    """
    MATCH_FEE_PERIOD: Final[str] = "fee_period"
    MATCH_START: Final[str] = "start"
    MATCH_END: Final[str] = "end"

    for line in lines:
        if line.startswith("Fee period "):
            match = re_fee_period.search(line)
            if match:
                return FeePeriodInfo(
                    full=match.group(MATCH_FEE_PERIOD),
                    start=match.group(MATCH_START),
                    end=match.group(MATCH_END)
                )
    return None


def extract_account_summary(
    lib_plumber_pdf: pdfplumber.pdf.PDF,
    lib_plumber_page: pdfplumber.page.Page,
    lib_mu_doc: pymupdf.Document,
    lib_mu_page: pymupdf.Page,
    lines: List[str],
    text: str,
    statement: Statement
) -> None:
    """
    The account summaries happen before the transactions.

    Results are stored in the statement class.

    Args:
        lines (List[str]): PDF as lines of text
        text (str): PDF as a single line of text separated by \n
        statement (Statement): Statement info
    """

    # Finding an account summary indicates we have found a new account
    account: Account = Account()
    statement.accounts.append(account)

    # Extract account name. The name has a line above and below. It one of the formats below:
    #       ------------------------------------
    #       Wells Fargo Money Market Savings SM
    #       ------------------------------------
    #       Statement period activity summary
    #
    #       ------------------------------------
    #       Wells Fargo Money Market Savings SM
    #       ------------------------------------
    #       Activity summary
    #
    # Extracting with regex is error prone, so the code below uses the lines to determine
    #   its location.
    raw_text_blocks: List[Tuple[float, float, float, float, str, int, int]] = lib_mu_page.get_text("blocks")
    text_blocks: List[str] = [t[4].replace('\n', ' ').strip() for t in raw_text_blocks]
    del raw_text_blocks

    for idx, txt in enumerate(text_blocks):
        if "Statement period activity summary" in txt or "Activity summary" in txt:
            tmp = text_blocks[idx - 1]
            if len(tmp) > 0:
                account.account_type = tmp
                break
            tmp = text_blocks[idx - 2]
            if len(tmp) > 0:
                account.account_type = tmp
                break

    if account.account_type is None:
        # We could not locate the name of the account type name
        account.account_type = "Could not locate"

    # Extract beginning balance and date
    begin_match = re.search(
        r"(?P<full_date>Beginning balance on (?P<begin_date>\d{1,2}/\d{1,2}))\s+\${0,1}(?P<begin_balance>\d{1,3}(?:,\d{3})*\.\d{2})",
        text,
        flags=re.MULTILINE
    )
    assert begin_match
    if begin_match:
        account.beginning_date = begin_match.group('full_date')
        account.beginning_date_mmyy = begin_match.group("begin_date")
        account.beginning_balance = begin_match.group("begin_balance")
        text = text[begin_match.end():]

    # Extract Deposits/Additions
    deposits_match = re.search(
        r"Deposits/Additions\s+\${0,1}?(?P<deposits>\d{1,3}(?:,\d{3})*\.\d{2})",
        text,
        flags=re.MULTILINE
    )
    assert deposits_match
    if deposits_match:
        account.total_deposits = deposits_match.group("deposits")
        text = text[deposits_match.end():]

    # Extract Withdrawals/Subtractions
    withdrawals_match = re.search(
        r"Withdrawals/Subtractions\s+-\s*\${0,1}(?P<withdrawals>\d{1,3}(?:,\d{3})*\.\d{2})",
        text,
        flags=re.MULTILINE
    )
    assert withdrawals_match
    if withdrawals_match:
        account.total_withdrawls = withdrawals_match.group("withdrawals")
        text = text[withdrawals_match.end():]

    # Extract ending balance and date
    end_match = re.search(
        r"(?P<full_date>Ending balance on (?P<end_date>\d{1,2}/\d{1,2}))\s*\${0,1}(?P<end_balance>\d{1,3}(?:,\d{3})*\.\d{2})",
        text,
        re.MULTILINE
    )
    assert end_match
    if end_match:
        account.ending_date = end_match.group('full_date')
        account.ending_date_mmyy = end_match.group("end_date")
        account.ending_balance = end_match.group("end_balance")
        text = text[end_match.end():]

    # Extract account number and owner name
    acc_number_match = re.search(
        r"Account number: \s+(?P<acc_number>\d{10})\s*\n(?P<owner>[A-Z ]+)\n",
        text,
        flags=re.MULTILINE
    )
    assert acc_number_match
    if acc_number_match:
        account.account_num = acc_number_match.group("acc_number")
        account.account_owner = acc_number_match.group("owner")
        text = text[acc_number_match.end():]


def calculate_vertical_column_separators(
    lib_plumber_page: pdfplumber.page.Page
) -> List[float]:
    """
    PdfPlumber's extract tables function assumes a table looks like this.
        +------------+------------+------------+------------+
        |    Date    |   Descr    | Deposits   | Withdrawls |
        +------------+------------+------------+------------+
        |            |            |            |            |
        +------------+------------+------------+------------+
        |            |            |            |            |
        +------------+------------+------------+------------+

    However, WellsFargo tables look like this
        -----------------------------------------------------
             Date        Descr      Deposits     Withdrawls
        -----------------------------------------------------

        -----------------------------------------------------

        -----------------------------------------------------

    So we need to manually specify the vertical lines using the
    names of the columns.

    Args:
        page (pdfplumber.page.Page): PdfPlumber page

    Returns:
        List[float]: List of X positions of the vertical
            column separators. Includes the leading and
            ending vertical poisitions.
    """

    explicit_verticals: List[float] = []
    date_pos: Dict[str, Any] = lib_plumber_page.search("Date")
    check_pos: Dict[str, Any] = lib_plumber_page.search("Number")
    desc_pos: Dict[str, Any] = lib_plumber_page.search("Description")
    deposits_pos: Dict[str, Any] = lib_plumber_page.search("Deposits/")
    withdrawls_pos: Dict[str, Any] = lib_plumber_page.search("Withdrawals/")
    subtractions_pos: Dict[str, Any] = lib_plumber_page.search("Subtractions")
    balance_pos: Dict[str, Any] = lib_plumber_page.search("Ending daily")
    x: float

    # Retrieve the lines to determine the leftmost and rightmost edges of the table.
    line = lib_plumber_page.lines[0]
    # Set the rightmost edge
    explicit_verticals.append(line['x0'])

    # Between Date and Check. Add +7 because the kerning for XX/XX
    #   may cause the vertical line to cut into the date
    x = date_pos[0]['x1'] + 7
    explicit_verticals.append(x)

    # Between Check and Descr
    # Some accounts don't have checking so always use the description position
    x = desc_pos[0]['x0']
    explicit_verticals.append(x)

    # Between Descr and Deposits. Large deposits need more room.
    x = deposits_pos[0]['x0'] - 20
    explicit_verticals.append(x)

    # Between Deposits and Withdrawls
    x = deposits_pos[0]['x1']
    explicit_verticals.append(x)

    # Between Withdrawls and Balance
    x = max(withdrawls_pos[0]['x1'], subtractions_pos[0]['x1'])
    explicit_verticals.append(x)

    # Set the leftmost edge
    explicit_verticals.append(line['x1'])

    return explicit_verticals


def extract_transaction_history(
    lib_plumber_pdf: pdfplumber.pdf.PDF,
    lib_plumber_page: pdfplumber.page.Page,
    lib_mu_doc: pymupdf.Document,
    lib_mu_page: pymupdf.Page,
    lines: List[str],
    text: str,
    statement: Statement
) -> None:
    global args
    # Locate the transaction history sections
    table_rects: List[pymupdf.Rect] = []
    table_title_rects: List[pymupdf.Rect] = lib_mu_page.search_for("Transaction history")
    table_footer_rect: List[pymupdf.Rect] = lib_mu_page.search_for("Ending balance on ")

    x1_pos_rightmost_column: float = lib_mu_page.rect.x1

    # We know we have a table, so error if we couldn't find it.
    assert len(table_title_rects) > 0

    prev_title_rect: pymupdf.Rect | None = None
    for idx, title_rect in enumerate(table_title_rects):
        cur_table_rect: pymupdf.Rect = pymupdf.Rect()
        table_rects.append(cur_table_rect)

        # Assign the bottom right of the title to the top left of the table rect
        cur_table_rect.x0 = title_rect.x0
        cur_table_rect.y0 = cur_table_rect.y1 = title_rect.y1
        cur_table_rect.x1 = x1_pos_rightmost_column

        # If we have a previous table title on the page, set the top of the
        #   current table to the bottom of the previous title.
        # We will refine the rectangle later in the algorithm.
        if prev_title_rect:
            cur_table_rect.y1 = prev_title_rect.y0
        else:
            cur_table_rect.y1 = lib_mu_page.rect.y1

        # prev_table_rect = cur_table_rect
        prev_title_rect = title_rect

    # We currently have estimated rectangles for each table.
    # We further refine the rectangle by attempting to locate the
    #   table footer "Ending balance on ".
    # And see if the end of the table is inside the
    #   estimated table rectangle.
    for footer_rect in table_footer_rect:
        for table_rect in table_rects:
            if footer_rect.intersects(table_rect):
                # Set the bottom of the table to the top of the footer.
                table_rect.y1 = footer_rect.y0

    # We should not have accurate table rectangles.
    # Try to extract the table data.
    extracted_tables: List[Any] = []
    for table_rect in table_rects:
        # Use PdfPlumber to extract the table data by specifying the rect

        cropped_page_pdf_plumber = lib_plumber_page.crop(table_rect)

        explicit_verticals = calculate_vertical_column_separators(cropped_page_pdf_plumber)
        # Options used by the "extract_table" method in PdfPlumber.
        extraction_options: Dict[str, Any] = {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "lines_strict",
            "explicit_vertical_lines": explicit_verticals
        }

        if args.debug:
            img = cropped_page_pdf_plumber.to_image(
                resolution=300  # ppi
            )
            img.debug_tablefinder(extraction_options)  # Overlay detected table structures
            img.show()
            img.save("debug.png", format="PNG")  # Save as temporary image

        extracted_table = cropped_page_pdf_plumber.extract_table(extraction_options)
        extracted_tables.append(extracted_table)

    # Store the transactions
    transaction_list: List[Transaction] = statement.accounts[-1].transaction_history
    for table in extracted_tables:
        for line in table:
            transaction: Transaction = Transaction()
            transaction_list.append(transaction)
            transaction.debug_str = line
            transaction.date = line[0] if line[0] != '' else None
            transaction.check_number = line[1] if line[1] != '' else None
            # Long descriptions that span two lines will have '\n" embedded.
            # Replace '\n' with a space
            transaction.description = line[2].replace('\n', ' ') if line[2] != '' else None
            transaction.deposits = line[3] if line[3] != '' else None
            transaction.withdrawls = line[4] if line[4] != '' else None
            transaction.daily_balance = line[5] if line[5] != '' else None

            sep_pos: int = transaction.date.index('/')
            assert sep_pos >= 0
            transaction.month = int(transaction.date[:sep_pos])
            transaction.day = int(transaction.date[sep_pos + 1:])


def convert_transactions_to_dataframe(
    statement: Statement
) -> None:
    """
    Proccess the transaction history and put the data
    into a DataFrame for numeric processing.

    Args:
        statement (Statement): Satement
    """
    # Get the beginning date of the statement
    beginning_date: dt.date = dt.datetime.strptime(
        statement.start_date,
        "%m/%d/%Y"
    ).date()

    all_dataframes: List[pd.DataFrame] = []
    for account in statement.accounts:
        # Shape the data
        #
        # pd.DataFrame.from_dict expects the data to be in the following format:
        #   {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
        #
        # The date column only lists Month/Day: 1/1 ... 12/31
        # Convert the date to include the year. Be careful to detect the year
        #   rollover.
        prev_month: int = beginning_date.month
        curr_year: int = beginning_date.year
        data: Dict[int, List[str | None]] = {}
        running_balance: float = float(account.beginning_balance.replace(",", ""))

        for idx, transaction in enumerate(account.transaction_history):
            assert transaction.month
            assert transaction.day

            curr_month = transaction.month
            # Did we move to the next year?
            #   12 > 1 we moved to the next year.
            if prev_month > curr_month:
                curr_year += 1

            deposits: float = float(transaction.deposits.replace(",", "")) if transaction.deposits else 0.0
            withdrawls: float = float(transaction.withdrawls.replace(",", "")) if transaction.withdrawls else 0.0
            running_balance = round(running_balance + deposits - withdrawls, 2)

            data[idx] = [
                dt.date(curr_year, curr_month, transaction.day),
                round(deposits, 2),
                round(withdrawls, 2),
                transaction.daily_balance,
                running_balance,
                transaction.check_number,
                account.account_num,
                account.account_type,
                transaction.description,
                statement.file_name,
                statement.file_path,
                statement.start_date,
                statement.end_date,
                statement.fee_period_dates,
                statement.statement_date,
                transaction.debug_str
            ]

            prev_month = curr_month

        monies: pd.DataFrame = pd.DataFrame.from_dict(
            data,
            orient='index',
            columns=[
                DATE,
                DEPOSITS,
                WITHDRAWLS,
                DAILY_BALANCE,
                RUNNING_BALANCE,
                CHECK_NUMBER,
                ACCOUNT_NUMBER,
                ACCOUNT_TYPE,
                DESCRIPTION,
                FILE_NAME,
                FILE_PATH,
                START_DATE,
                END_DATE,
                FEE_PERIOD_DATES,
                STATEMENT_DATE,
                DEBUG_STR
            ],
            dtype=str
        )

        monies[DEPOSITS] = monies[DEPOSITS].str.replace(",", "").astype(float).fillna(0)
        monies[WITHDRAWLS] = monies[WITHDRAWLS].str.replace(",", "").astype(float).fillna(0)
        monies[DAILY_BALANCE] = monies[DAILY_BALANCE].str.replace(",", "").astype(float)
        # monies[CHECK_NUMBER] = monies[CHECK_NUMBER].str.replace(",", "").astype(int)

        account.df_transactions = monies
        all_dataframes.append(monies)

    statement.df_transactions_combined = pd.concat(all_dataframes, ignore_index=True)  # reset the index ignore_index=True


def convert_pdf(pdf_path: str) -> Statement:
    """
    Extracts transaction history from a Wells Fargo statement PDF.
    Also extracts the summary data which is used to verify the
    data was extracted successfully.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        Statement: Statement class
    """

    # Create a new statement for this file.
    statement: Statement = Statement(
        file_path=pdf_path,
        file_name=os.path.basename(pdf_path)
    )

    # #####################################################################
    # Extracting the summary data and misc information is done
    #   using regular expressions.
    # Extracting the transaction data is harder and we use
    #   PdfPlumber's extract table methods.
    # We use two separate libraries since each makes some things
    #   easier than the other library.
    # #####################################################################

    # When we open a file we need to get the statement date. This flag
    #   indicates if it needs to be extracted.
    extract_date_pending: bool = True
    extract_fee_period_pending: bool = True
    with pdfplumber.open(pdf_path) as lib_plumber_pdf:
        with pymupdf.open(pdf_path) as lib_mu_pdf:
            for lib_mu_page in lib_mu_pdf:
                lib_plumber_page: pdfplumber.page.Page = lib_plumber_pdf.pages[lib_mu_page.number]
                assert lib_plumber_page

                # Convert the PDF page to text. Text blocks are spearated by \n
                text: str = cast(str, lib_mu_page.get_text())
                # Get the individual lines of text
                lines: List[str] = [line.strip() for line in text.split("\n")]

                if extract_date_pending:
                    extract_date_pending = False
                    # Extract the statement date
                    statement.statement_date = extract_date(lines)

                if extract_fee_period_pending:
                    # Extract the fee period
                    fee_period: FeePeriodInfo | None = extract_fee_period(lines)
                    if fee_period:
                        extract_fee_period_pending = False
                        statement.fee_period_dates = fee_period.full
                        statement.start_date = fee_period.start
                        statement.end_date = fee_period.end

                # Scan the page and determine the order of the different sections
                pdf_sections = determine_processing_order(text)

                # The order is important. Summaries occur before transactions.
                # Finding a summary section means we have found a new account.
                for section in pdf_sections:
                    if PDF_SECTION_STATEMENT_SUMMARY == section.pdf_section:
                        extract_account_summary(
                            lib_plumber_pdf,
                            lib_plumber_page,
                            lib_mu_pdf,
                            lib_mu_page,
                            lines,
                            text,
                            statement)
                    elif PDF_SECTION_TRANSACTIONS == section.pdf_section:
                        extract_transaction_history(
                            lib_plumber_pdf,
                            lib_plumber_page,
                            lib_mu_pdf,
                            lib_mu_page,
                            lines,
                            text,
                            statement
                        )

    # Convert the transactions to a DataFrame
    convert_transactions_to_dataframe(statement)

    # Do some simple validation to make sure all of the fields were properly filled.
    validate_statement(statement)
    return statement


def batch_convert(directory: str) -> List[Statement]:
    """
    Convert a folder with PDF files

    Args:
        directory (str): Folder path

    Returns:
        List[Statement]: List of statements
    """
    statements: List[Statement] = []

    for root, dirs, files in os.walk(directory):  # type: ignore
        for file in files:
            if file.endswith(".pdf"):
                statements.append(convert_pdf(os.path.join(root, file)))

    return statements


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Show the table extraction visually.", action="store_true")
    parser.add_argument("-nc", "--nocsv", help="Do not create a CSV output file.", action="store_true")
    parser.add_argument("-nj", "--nojson", help="Do not create a Parquet output file.", action="store_true")
    parser.add_argument("-np", "--noparquet", help="Do not create a Parquet output file.", action="store_true")
    parser.add_argument("-ne", "--noexcel", help="Do not create an Excel output file.", action="store_true")
    parser.add_argument("-o", "--outputpath", help="Ouput path", default=os.getcwd())
    parser.add_argument("paths", help="List of paths to a PDF file or directory.", nargs="+")
    args = parser.parse_args()

    statements: List[Statement] = []

    for pdf_path in args.paths:
        if os.path.isdir(pdf_path):
            statements.extend(batch_convert(pdf_path))
        else:
            if not pdf_path.endswith(".pdf"):
                continue
            statements.append(convert_pdf(pdf_path))

    list_all_dfs: List[pd.DataFrame] = []
    for statement in statements:
        list_all_dfs.append(statement.df_transactions_combined)
        file_path = statement.file_path[:-4]

        if args.nocsv is False:
            statement.df_transactions_combined.to_csv(file_path + '.csv')
        if args.noexcel is False:
            statement.df_transactions_combined.to_excel(file_path + '.xlsx')
        if args.noparquet is False:
            statement.df_transactions_combined.to_parquet(file_path + '.parquet')
        if args.nojson is False:
            statement.df_transactions_combined.to_json(file_path + '.json')
            # with open(output_path + '-debug.json', "w", encoding='UTF-8') as f:
            #     json.dumps(statement.to_json(), f, cls=DataclassEncoder, indent=4)
    df_all: pd.DataFrame = pd.concat(list_all_dfs, ignore_index=True)

    if args.nocsv is False:
        df_all.to_csv(os.path.join(args.outputpath, 'all_combined.csv'))
    if args.noexcel is False:
        df_all.to_excel(os.path.join(args.outputpath, 'all_combined.xlsx'))
    if args.noparquet is False:
        df_all.to_parquet(os.path.join(args.outputpath, 'all_combined.parquet'))
    if args.nojson is False:
        df_all.to_json(os.path.join(args.outputpath, 'all_combined.json'))


main()
