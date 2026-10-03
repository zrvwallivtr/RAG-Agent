import socket


def _is_connected(host="1.1.1.1", port=53, timeout=3) -> bool:
    """
    Returns True if the system can connect to the host/port,
    otherwise returns false.
    Host (Cloudflare DNS):  1.1.1.1
    Port (DNS traffic):     53
    """
    try:
        # Create socket object with connection timeout
        socket.setdefaulttimeout(timeout)

        # Attempt to connect to the host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True

    except (socket.timeout, OSError):
        return False
