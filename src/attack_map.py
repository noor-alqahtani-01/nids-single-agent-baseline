"""Tool 2: map a CICIDS2017 attack label to a MITRE ATT&CK technique."""

ATTACK_MAP = {
    "FTP-Patator": ("T1110", "Brute Force"),
    "SSH-Patator": ("T1110", "Brute Force"),
    "DoS Hulk": ("T1499", "Endpoint Denial of Service"),
    "DoS GoldenEye": ("T1499", "Endpoint Denial of Service"),
    "DoS slowloris": ("T1499", "Endpoint Denial of Service"),
    "DoS Slowhttptest": ("T1499", "Endpoint Denial of Service"),
    "DDoS": ("T1498", "Network Denial of Service"),
    "Heartbleed": ("T1190", "Exploit Public-Facing Application"),
}


def lookup_attack(attack_type: str) -> dict:
    """Return the ATT&CK technique for a predicted attack type."""
    if attack_type == "BENIGN":
        return {"technique_id": None, "technique_name": None}
    tid, name = ATTACK_MAP.get(attack_type, ("unknown", "Unmapped technique"))
    return {"technique_id": tid, "technique_name": name}
