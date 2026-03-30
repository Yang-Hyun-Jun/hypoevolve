1. 탐색 과정 하나에서는 시간은 모두 time step 개념으로 통일하기 (o)
2. Measurable 변환에서 Measure 표현 description 같이 만들어서 evaluation 프롬프트에 정보 주기
4. Muation Steering 할 때 ELG의 자연어 표현도 같이 받기 (o)
5. 웬만한 주요 정보는 시스템 프롬프트에 넣기
6. 프롬프트 최적화 필요
7. 리프 노드 자유 생성
8. ELG 자연어 만들면 재생성하지 말고 저장해두고 샘플링 했을 때 그 정보 그대로 쓰기 (o)



  ### 1. locality가 사라질 수 있음

  parent를 mutation하는 게 아니라 아예 unrelated hypothesis를 던질 수 있음.

  ### 2. measurable discipline이 깨질 수 있음

  현재 evaluator는 사실상 measurable ELG를 기대하는데,
  자유 생성하면 semantic하게 흐를 위험이 큼.

  ### 3. invalid JSON / invalid ELG / degenerate tree

  LLM이 구조를 자주 깨먹을 수 있음.

  ### 4. archive duplication / mode collapse

  조금만 다른 표현, 혹은 같은 fingerprint 반복 생성 가능.

  ### 5. complexity explosion

  트리가 무한히 커질 수 있음.