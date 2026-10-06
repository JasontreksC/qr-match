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


def tag_inclusion_min(tagA_want: list[str], tagA_have: list[str], tagB_want: list[str], tagB_have: list[str]) -> float:
    """양방향 포함 계수(|want ∩ 상대 have| / |want|) 중 작은 값. 최종 점수 동점 처리에만 쓴다.

    tag_score의 계산 방식은 바꾸지 않는다. 이 함수는 같은 입력에서 방향별 값을 따로 꺼내 보기만 한다.
    """
    coefficients = []
    for want, have in ((tagA_want, tagB_have), (tagB_want, tagA_have)):
        wanted = set(want)
        coefficients.append(len(wanted & set(have)) / len(wanted) if wanted else 0.0)
    return min(coefficients)
