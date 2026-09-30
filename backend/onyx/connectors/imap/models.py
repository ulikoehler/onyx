import email.header
import email.utils
import hashlib
from datetime import datetime
from datetime import timezone
from email.message import Message
from enum import Enum

from pydantic import BaseModel


class Header(str, Enum):
    SUBJECT_HEADER = "subject"
    FROM_HEADER = "from"
    TO_HEADER = "to"
    DELIVERED_TO_HEADER = (
        "Delivered-To"  # Used in mailing lists instead of the "to" header.
    )
    DATE_HEADER = "date"
    MESSAGE_ID_HEADER = "Message-ID"


class EmailHeaders(BaseModel):
    """
    Model for email headers extracted from IMAP messages.
    """

    id: str
    subject: str
    sender: str
    recipients: str | None
    date: datetime

    @classmethod
    def from_email_msg(cls, email_msg: Message) -> "EmailHeaders":
        def _decode(header: str, default: str | None = None) -> str | None:
            value = email_msg.get(header, default)
            if not value:
                return None

            # A header can consist of multiple encoded segments; decode all of
            # them and join, otherwise trailing content (e.g. the <address>
            # part) is silently dropped.
            segments: list[str] = []
            for decoded_value, encoding in email.header.decode_header(value):
                if isinstance(decoded_value, bytes):
                    encoding = encoding or "utf-8"
                    try:
                        segments.append(
                            decoded_value.decode(encoding, errors="replace")
                        )
                    except LookupError:
                        # decode_header returns pseudo-charsets like
                        # "unknown-8bit" for raw 8-bit data; fall back to utf-8.
                        segments.append(
                            decoded_value.decode("utf-8", errors="replace")
                        )
                elif isinstance(decoded_value, str):
                    segments.append(decoded_value)
            return "".join(segments)

        def _parse_date(date_str: str | None) -> datetime | None:
            if not date_str:
                return None
            try:
                return email.utils.parsedate_to_datetime(date_str)
            except (TypeError, ValueError):
                return None

        message_id = _decode(header=Header.MESSAGE_ID_HEADER)
        if not message_id:
            # Deterministic fallback id so messages without Message-ID can
            # still be indexed.
            message_id = "no-message-id-" + hashlib.sha256(
                email_msg.as_bytes()
            ).hexdigest()
        # It's possible for the subject line to not exist or be an empty string.
        subject = _decode(header=Header.SUBJECT_HEADER) or "Unknown Subject"
        from_ = _decode(header=Header.FROM_HEADER) or ""
        to = _decode(header=Header.TO_HEADER)
        if not to:
            to = _decode(header=Header.DELIVERED_TO_HEADER)
        date_str = _decode(header=Header.DATE_HEADER)
        date = _parse_date(date_str=date_str) or datetime.fromtimestamp(
            0, tz=timezone.utc
        )
        return cls.model_validate(
            {
                "id": message_id,
                "subject": subject,
                "sender": from_,
                "recipients": to,
                "date": date,
            }
        )
