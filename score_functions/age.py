def age_accepts(my_age: int, their_age: int, want: list[str]) -> bool:
    if "상관없음" in want:
        return True
    if my_age == their_age:
        return "동갑" in want
    if my_age > their_age:
        return "연하" in want
    return "연상" in want