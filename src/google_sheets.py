import pandas as pd
import requests
import re
from datetime import datetime


def extract_sheet_id(url: str) -> str | None:
    pattern = r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def extract_gid(url: str) -> str:
    match = re.search(r"gid=(\d+)", url)
    return match.group(1) if match else "0"


def build_csv_export_url(sheet_id: str, gid: str = "0") -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def get_sheet_name(sheet_id: str) -> str:
    """Try to fetch the sheet title from Google Sheets HTML page."""
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        response = requests.get(url, timeout=10)
        match = re.search(r"<title>(.*?)</title>", response.text)
        if match:
            title = match.group(1).replace(" - Google Sheets", "").strip()
            return title
    except:
        pass
    return "Unknown"


def clean_url(url: str) -> str:
    """
    Strips extra query parameters from Google Sheets URLs.
    Also handles Excel-format URLs (rtpof=true) by forcing CSV export.
    """
    url = url.strip()
    gid_match = re.search(r"gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    sheet_id_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not sheet_id_match:
        return url
    sheet_id = sheet_id_match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid={gid}"


def is_excel_format(url: str) -> bool:
    """Detects if the URL points to an Excel file stored in Drive."""
    return "rtpof=true" in url or "sd=true" in url


def load_google_sheet(url: str) -> dict:
    """
    Loads a public Google Sheet into a pandas DataFrame.
    Returns a dict with: success, data, message, rows, sheet_name, loaded_at
    """
    result = {
        "success":    False,
        "data":       None,
        "message":    "",
        "rows":       0,
        "sheet_name": "Unknown",
        "loaded_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Detect Excel format and warn user but still try
    excel_format = is_excel_format(url)

    # Clean the URL first
    url = clean_url(url)

    if "docs.google.com/spreadsheets" not in url:
        result["message"] = "This does not look like a Google Sheets URL. Please paste the full share link."
        return result

    sheet_id = extract_sheet_id(url)
    if not sheet_id:
        result["message"] = "Could not find the Sheet ID in the URL. Make sure you copied the full link."
        return result

    gid     = extract_gid(url)
    csv_url = build_csv_export_url(sheet_id, gid)

    try:
        response = requests.get(csv_url, timeout=15)

        if response.status_code != 200:
            # If Excel format, try xlsx export instead
            if excel_format:
                sheet_id = extract_sheet_id(url)
                xlsx_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
                try:
                    xlsx_response = requests.get(xlsx_url, timeout=15)
                    if xlsx_response.status_code == 200:
                        from io import BytesIO
                        df = pd.read_excel(BytesIO(xlsx_response.content))
                        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
                        df.columns = df.columns.str.strip()
                        result["success"]    = True
                        result["data"]       = df
                        result["rows"]       = len(df)
                        result["sheet_name"] = get_sheet_name(sheet_id)
                        result["loaded_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result["message"]    = f"✅ Sheet loaded — {len(df):,} rows × {len(df.columns)} columns"
                        return result
                except:
                    pass
            result["message"] = (
                f"Could not access the sheet (HTTP {response.status_code}). "
                "Make sure the sheet is set to 'Anyone with the link can view'."
            )
            return result

        if response.text.strip().startswith("<!DOCTYPE") or \
           response.text.strip().startswith("<html"):
            result["message"] = (
                "The sheet appears to be private. "
                "In Google Sheets go to Share → Change to Anyone with the link → Viewer."
            )
            return result

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        df.columns = df.columns.str.strip()

        # Try to get sheet name
        sheet_name = get_sheet_name(sheet_id)

        result["success"]    = True
        result["data"]       = df
        result["rows"]       = len(df)
        result["sheet_name"] = sheet_name
        result["loaded_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["message"]    = f"✅ Sheet loaded — {len(df):,} rows × {len(df.columns)} columns"

    except requests.exceptions.Timeout:
        result["message"] = "Request timed out. Check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        result["message"] = "Could not connect. Check your internet connection."
    except Exception as e:
        result["message"] = f"Unexpected error: {str(e)}"

    return result