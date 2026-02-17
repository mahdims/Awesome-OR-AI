"""
Layer 3: Email Sender

Sends HTML emails via Gmail SMTP using a Google App Password.
Falls back to saving HTML files if credentials are not configured.

Required environment variables:
    GMAIL_USER         — your Gmail address (sender + auth)
    GMAIL_APP_PASSWORD — 16-char Google App Password
    EMAIL_TO           — comma-separated recipient addresses
"""

import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional, Union


_GMAIL_HOST = "smtp.gmail.com"
_GMAIL_PORT = 587


def send_email(to: Union[str, List[str]], subject: str, html_content: str) -> bool:
    """
    Send an HTML email via Gmail SMTP.

    Args:
        to:           Single address or list of recipient addresses.
        subject:      Email subject line.
        html_content: Full HTML email body.

    Returns:
        True if sent successfully, False otherwise.
    """
    gmail_user = os.getenv("GMAIL_USER")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not app_password:
        print("[WARN] GMAIL_USER or GMAIL_APP_PASSWORD not set. Cannot send email.")
        return False

    recipients = [to] if isinstance(to, str) else to

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(_GMAIL_HOST, _GMAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, app_password)
            server.sendmail(gmail_user, recipients, msg.as_string())

        print(f"[OK] Email sent to {', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Gmail authentication failed. Check GMAIL_USER and GMAIL_APP_PASSWORD.")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


def save_html(html_content: str, output_path: Path) -> Path:
    """
    Save HTML email to a local file for preview/archive.

    Args:
        html_content: Full HTML email body.
        output_path:  Path to save the HTML file.

    Returns:
        The path where the file was saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[OK] Email saved to {output_path}")
    return output_path


def deliver(html_content: str, subject: str,
            preview_dir: Optional[Path] = None,
            filename: Optional[str] = None,
            send: bool = True) -> bool:
    """
    Deliver email: send via Gmail and/or save locally.

    EMAIL_TO env var supports comma-separated addresses.

    Args:
        html_content: Full HTML body.
        subject:      Email subject.
        preview_dir:  Directory to save HTML file (None = skip saving).
        filename:     Filename for saved HTML (default: auto-generated).
        send:         Whether to attempt sending via Gmail.

    Returns:
        True if at least one delivery method succeeded.
    """
    success = False

    # Save to file
    if preview_dir:
        if filename is None:
            slug = subject.lower().replace(" ", "_")[:40]
            filename = f"{slug}_{date.today().isoformat()}.html"
        path = preview_dir / filename
        save_html(html_content, path)
        success = True

    # Send via Gmail
    if send:
        to_raw = os.getenv("EMAIL_TO", "")
        recipients = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

        if recipients:
            sent = send_email(recipients, subject, html_content)
            success = success or sent
        else:
            print("[WARN] EMAIL_TO not set. Skipping email delivery.")
            if not preview_dir:
                # Fallback: save to default location so output isn't lost
                fallback_dir = (
                    Path(__file__).parent.parent.parent
                    / "docs" / "living_reviews" / "emails"
                )
                if filename is None:
                    slug = subject.lower().replace(" ", "_")[:40]
                    filename = f"{slug}_{date.today().isoformat()}.html"
                save_html(html_content, fallback_dir / filename)
                success = True

    return success
