def mbti_score(mbti_a:str, mbti_b:str) -> float: 
    mbti_a = mbti_a.upper()
    mbti_b = mbti_b.upper()
    score = 0
    MAX = 4
    if len(mbti_a) != MAX or len(mbti_b) != MAX:
        print("MBTI는 네자리 문자열로 입력해야 합니다.")
        return -1

    if not mbti_a[0] in ("I" "E") or not mbti_b[0] in ("I", "E"):
        print("MBTI 첫번째는 I 또는 E 여야 합니다.")
        return -1
    if not mbti_a[1] in ("N" "S") or not mbti_b[1] in ("N", "S"):
        print("MBTI 첫번째는 N 또는 S 여야 합니다.")
        return -1
    if not mbti_a[2] in ("F" "T") or not mbti_b[2] in ("F", "T"):
        print("MBTI 첫번째는 F 또는 T 여야 합니다.")
        return -1
    if not mbti_a[3] in ("P" "J") or not mbti_b[3] in ("P", "J"):
        print("MBTI 첫번째는 P 또는 J 여야 합니다.")
        return -1


    
    for i in range(MAX):
        if i in (0,3):
            if mbti_a[i] != mbti_b[i]:
                score += 1
        else:
            if mbti_a[i] == mbti_b[i]:
                score += 1
        
    
    score /= 4
    return score