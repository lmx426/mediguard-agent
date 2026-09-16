"""Helpers for Shanghai drug price API encrypted request files.

The H5 app uses a browser-only flow: fetch an RSA public key, encrypt a random
AES key with RSA, then AES-ECB encrypt the business request. This helper keeps
network access outside Python so the Windows sandbox can call curl.exe directly.
It only prepares request bodies and decrypts response files.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import secrets
import string
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


SHANGHAI_TZ = timezone(timedelta(hours=8))
DEFAULT_SERVICE_BASE = "https://bjxt.smiic.net.cn/hsa-mbs-pub/api/v1/main"
DEFAULT_SOURCE_URL = "https://bjxt.smiic.net.cn/ypcx/?sessionid=#/pages/drugs-query/search"
RAW_SCHEMA_VERSION = "shanghai_drug_price_raw_v0.1"
ACCESS_TOKEN_RE = re.compile(r'accessToken\s*:\s*"(?P<token>[^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    session_parser = subparsers.add_parser("prepare-session")
    session_parser.add_argument("--keys-response", type=Path, required=True)
    session_parser.add_argument("--common-js", type=Path, required=True)
    session_parser.add_argument("--session-output", type=Path, required=True)
    session_parser.add_argument("--service-base", default=DEFAULT_SERVICE_BASE)

    request_parser = subparsers.add_parser("prepare-request")
    request_parser.add_argument("--session", type=Path, required=True)
    request_parser.add_argument("--keyword", required=True)
    request_parser.add_argument("--page-num", type=int, required=True)
    request_parser.add_argument("--page-size", type=int, required=True)
    request_parser.add_argument("--request-output", type=Path, required=True)
    request_parser.add_argument("--service-code", default="YBDRUG001")

    decode_parser = subparsers.add_parser("decode-response")
    decode_parser.add_argument("--session", type=Path, required=True)
    decode_parser.add_argument("--response", type=Path, required=True)
    decode_parser.add_argument("--raw-output", type=Path, required=True)
    decode_parser.add_argument("--issue-output", type=Path, required=True)
    decode_parser.add_argument("--keyword", required=True)
    decode_parser.add_argument("--page-num", type=int, required=True)
    decode_parser.add_argument("--page-size", type=int, required=True)
    decode_parser.add_argument("--service-code", default="YBDRUG001")
    decode_parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-session":
        prepare_session(args)
    elif args.command == "prepare-request":
        prepare_request(args)
    elif args.command == "decode-response":
        decode_response(args)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(args.command)


def prepare_session(args: argparse.Namespace) -> None:
    keys_response = read_json(args.keys_response)
    if keys_response.get("code") != 0:
        raise RuntimeError(f"keys response failed: {keys_response}")
    data = keys_response.get("data") or {}
    client_id = str(data.get("clientId") or "").strip()
    rsa_public_key = str(data.get("rsaPublicKey") or "").strip()
    if not client_id or not rsa_public_key:
        raise RuntimeError("keys response is missing clientId or rsaPublicKey")

    access_token = extract_access_token(args.common_js)
    aes_key = random_aes_key()
    public_key = serialization.load_der_public_key(base64.b64decode(rsa_public_key))
    encrypted_key = public_key.encrypt(aes_key.encode("utf-8"), asym_padding.PKCS1v15())

    session = {
        "client_id": client_id,
        "aes_key": aes_key,
        "encrypted_aes_key": base64.b64encode(encrypted_key).decode("ascii"),
        "access_token": access_token,
        "service_base": str(args.service_base).rstrip("/"),
        "created_at": now_iso(),
    }
    write_json(args.session_output, session)
    print(
        json.dumps(
            {
                "status": "ok",
                "client_id": client_id,
                "session_output": str(args.session_output),
            },
            ensure_ascii=False,
        )
    )


def prepare_request(args: argparse.Namespace) -> None:
    session = read_json(args.session)
    payload = {
        "code": args.service_code,
        "identity": "",
        "departId": "",
        "purpose": "1",
        "qdlybz": "17",
        "admdvsL": "310000",
        "reqData": [
            {
                "drugName": args.keyword,
                "saleSortFlag": "",
                "pageSize": args.page_size,
                "pageNum": args.page_num,
            }
        ],
    }
    request_body = {
        "encryptedAESKey": session["encrypted_aes_key"],
        "encryptText": aes_encrypt(payload, session["aes_key"]),
        "pubChnl": "3",
        "accessToken": session["access_token"],
    }
    write_json(args.request_output, request_body)


def decode_response(args: argparse.Namespace) -> None:
    session = read_json(args.session)
    response = read_json(args.response)
    fetched_at = now_iso()
    period_start, period_end = previous_settlement_week(date.today())
    if response.get("code") != 0:
        append_jsonl(
            args.issue_output,
            {
                "keyword": args.keyword,
                "page_num": args.page_num,
                "fetched_at": fetched_at,
                "response": response,
            },
        )
        print(json.dumps({"status": "api_error", "keyword": args.keyword}, ensure_ascii=False))
        return

    encrypted_data = response.get("data")
    decoded = aes_decrypt(encrypted_data, session["aes_key"]) if encrypted_data else {}
    page_rows = decoded.get("data") if isinstance(decoded, dict) else None
    if not isinstance(page_rows, list):
        append_jsonl(
            args.issue_output,
            {
                "keyword": args.keyword,
                "page_num": args.page_num,
                "fetched_at": fetched_at,
                "decoded": decoded,
                "issue": "decoded response has no data list",
            },
        )
        page_rows = []

    page_meta = {
        key: value
        for key, value in (decoded or {}).items()
        if key != "data"
    } if isinstance(decoded, dict) else {}
    for row_index, row in enumerate(page_rows, start=1):
        append_jsonl(
            args.raw_output,
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "source_platform": "国家医保开放平台上海专区",
                "source_url": args.source_url,
                "source_service_code": args.service_code,
                "jurisdiction": "shanghai",
                "query_keyword": args.keyword,
                "page_num": args.page_num,
                "page_size": args.page_size,
                "row_index_in_page": row_index,
                "fetched_at": fetched_at,
                "price_period_label": "上周",
                "price_period_start": period_start.isoformat(),
                "price_period_end": period_end.isoformat(),
                "response_page": page_meta,
                "raw_row": row,
            },
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "keyword": args.keyword,
                "page_num": args.page_num,
                "row_count": len(page_rows),
            },
            ensure_ascii=False,
        )
    )


def extract_access_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = ACCESS_TOKEN_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not find public H5 accessToken in {path}")
    return match.group("token")


def random_aes_key(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def aes_encrypt(payload: Any, key: str) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padder = PKCS7(128).padder()
    padded = padder.update(text) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key.encode("utf-8")), modes.ECB()).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode("ascii")


def aes_decrypt(ciphertext: str, key: str) -> Any:
    decryptor = Cipher(algorithms.AES(key.encode("utf-8")), modes.ECB()).decryptor()
    padded = decryptor.update(base64.b64decode(ciphertext)) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    data = unpadder.update(padded) + unpadder.finalize()
    text = data.decode("utf-8")
    return json.loads(text) if text[:1] in {"{", "["} else text


def previous_settlement_week(today: date) -> tuple[date, date]:
    # The H5 page displays the previous Mon-Sun settlement week; on Mon/Tue it
    # shows the week before last, matching the bundled dayjs logic.
    js_day = (today.weekday() + 1) % 7
    base = today - timedelta(weeks=1 if js_day >= 2 or js_day == 0 else 2)
    base_js_day = (base.weekday() + 1) % 7
    day_index = base_js_day or 7
    start = base - timedelta(days=day_index - 1)
    end = base + timedelta(days=7 - day_index)
    return start, end


def now_iso() -> str:
    return datetime.now(SHANGHAI_TZ).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        file_obj.write("\n")


if __name__ == "__main__":
    main()
