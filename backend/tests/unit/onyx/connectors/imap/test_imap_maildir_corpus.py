"""Run the IMAP connector's parsing pipeline against a maildir corpus.

Set ONYX_TEST_MAILDIR to a maildir directory (or any directory of RFC822
files; searched recursively) and each message is fed through the same code
path the connector uses during a fetch:

    email.message_from_bytes
    -> EmailHeaders.from_email_msg
    -> _convert_email_headers_and_body_into_document

Every mail becomes one parametrized test, so a single malformed message
fails without hiding the results of the rest of the corpus.

Standalone usage (e.g. inside a container without pytest):

    python test_imap_maildir_corpus.py /path/to/maildir
"""

import os
import sys
from email import message_from_bytes
from email.message import Message
from pathlib import Path

try:
    import pytest
except ImportError:  # standalone __main__ mode without pytest installed
    pytest = None  # type: ignore[assignment]

from onyx.connectors.imap.connector import (
    _convert_email_headers_and_body_into_document,
)
from onyx.connectors.imap.models import EmailHeaders

_MAILDIR_ENV = "ONYX_TEST_MAILDIR"


def _iter_maildir_files(maildir: Path) -> list[Path]:
    return sorted(p for p in maildir.rglob("*") if p.is_file())


def _mail_files() -> list[Path]:
    maildir = os.environ.get(_MAILDIR_ENV, "").strip()
    if not maildir:
        return []
    return _iter_maildir_files(Path(maildir))


def _process_email(path: Path) -> None:
    email_msg: Message = message_from_bytes(path.read_bytes())
    email_headers = EmailHeaders.from_email_msg(email_msg=email_msg)
    _convert_email_headers_and_body_into_document(
        email_msg=email_msg,
        email_headers=email_headers,
        include_perm_sync=False,
    )


if pytest is not None:

    @pytest.mark.parametrize(
        "mail_path",
        _mail_files(),
        ids=lambda p: p.name,
    )
    def test_maildir_email_parses(mail_path: Path) -> None:
        _process_email(mail_path)

    def test_maildir_env_var_set() -> None:
        if not os.environ.get(_MAILDIR_ENV):
            pytest.skip(
                f"{_MAILDIR_ENV} not set; point it at a maildir to run the corpus"
            )


def _process_email_capturing(path: Path) -> tuple[Path, str | None]:
    try:
        _process_email(path)
        return path, None
    except Exception as e:  # noqa: BLE001 - corpus harness reports all failures
        return path, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <maildir>", file=sys.stderr)
        sys.exit(2)

    files = _iter_maildir_files(Path(sys.argv[1]))
    failures: dict[Path, str] = {}
    with ProcessPoolExecutor(max_workers=16) as pool:
        for path, error in pool.map(
            _process_email_capturing, files, chunksize=32
        ):
            if error is not None:
                failures[path] = error

    for path, error in failures.items():
        print(f"FAIL {path}: {error}")
    print(f"{len(files) - len(failures)}/{len(files)} emails parsed successfully")
    sys.exit(1 if failures else 0)
