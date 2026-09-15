def birth_accepts(my_birth: str | None, their_birth: str | None, want: list[str]) -> bool:
    if "상관없음" in want:
        return True
    mine = _parse_birth(my_birth)
    theirs = _parse_birth(their_birth)
    if mine is None or theirs is None:
        return False
    if mine[0] == theirs[0]:
        return "동갑" in want
    if mine < theirs:
        return "연하" in want
    return "연상" in want


def _parse_birth(raw: str | None) -> tuple[int, int, int] | None:
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if len(digits) != 6:
        return None
    yy = int(digits[0:2])
    mm = int(digits[2:4])
    dd = int(digits[4:6])
    if mm < 1 or mm > 12 or dd < 1 or dd > 31:
        return None
    from datetime import date

    today = date.today()
    year = 2000 + yy if yy <= today.year % 100 else 1900 + yy
    try:
        parsed = date(year, mm, dd)
    except ValueError:
        return None
    if parsed > today:
        return None
    return (parsed.year, parsed.month, parsed.day)
