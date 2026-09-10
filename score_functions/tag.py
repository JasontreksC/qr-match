def tag_score(tagA_want: list[str], tagA_have: list[str], tagB_want: list[str], tagB_have: list[str]) -> float:
    tagA_ws = set(tagA_want)
    tagA_hs = set(tagA_have)
    tagB_ws = set(tagB_want)
    tagB_hs = set(tagB_have)

    tag_AtoB = list(tagA_ws.intersection(tagB_hs))
    tag_BtoA = list(tagB_ws.intersection(tagA_hs))

    score_AtoB = len(tag_AtoB)/len(tagA_ws)
    score_BtoA = len(tag_BtoA)/len(tagB_ws)

    # 양쪽 교집합이 모두 없으면 조화평균 분모가 0이다. 태그 궁합 없음 = 0점.
    if score_AtoB + score_BtoA == 0:
        return 0.0
    
    final_score = 2 * score_AtoB * score_BtoA / (score_AtoB + score_BtoA)
    return final_score
