import os
import sys
import asyncio # 비동기 모듈

# 현재 파일의 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(__file__))

# main파일 로드
import main

# ─── 출력 구분선 헬퍼 ────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(label: str, value):
    print(f"  [OK] {label}: {value}")

def info(label: str, value):
    print(f"  [  ] {label}: {value}")

# 1. 환경 변수 출력
def demo_step1_env():
    section("STEP 1 — 환경변수 (ENV)")

    ok("TMDB_API_KEY", main.TMDB_API_KEY)
    ok("TMDB_BASE",    main.TMDB_BASE)
    ok("TMDB_IMG_500", main.TMDB_IMG_500)

    # TMDB_API_KEY 없을 때 RuntimeError 흐름 설명
    print()
    info("TMDB_API_KEY 없으면?", "→ RuntimeError('TMDB_API_KEY missing') 발생")

# 3. pydantic 모델 테스트
def demo_step2_models():
    section("STEP 2 — Pydantic 모델 (MODELS)")

    # TMDBMovieCard
    card = main.TMDBMovieCard(
        tmdb_id=550,
        title="Fight Club",
        poster_url="https://image.tmdb.org/t/p/w500/test.jpg",
        release_date="1999-10-15",
        vote_average=8.4,
    )
    print("\n  [TMDBMovieCard]")
    ok("tmdb_id",      card.tmdb_id)
    ok("title",        card.title)
    ok("poster_url",   card.poster_url)
    ok("release_date", card.release_date)
    ok("vote_average", card.vote_average)

    # TMDBMovieDetails
    details = main.TMDBMovieDetails(
        tmdb_id=550,
        title="Fight Club",
        overview="An insomniac office worker forms an underground fight club...",
        genres=[{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
    )
    print("\n  [TMDBMovieDetails]")
    ok("tmdb_id",  details.tmdb_id)
    ok("title",    details.title)
    ok("overview", details.overview[:40] + "...")
    ok("genres",   details.genres)

    # TFIDFRecItem
    item = main.TFIDFRecItem(title="Inception", score=0.87)
    print("\n  [TFIDFRecItem]")
    ok("title", item.title)
    ok("score", item.score)
    ok("tmdb",  item.tmdb)  # None (Optional)

    # SearchBundleResponse
    bundle = main.SearchBundleResponse(
        query="Inception",
        movie_details=details,
        tfidf_recommendations=[item],
        genre_recommendations=[card],
    )
    print("\n  [SearchBundleResponse]")
    ok("query",                  bundle.query)
    ok("movie_details.title",    bundle.movie_details.title)
    ok("tfidf_recommendations",  len(bundle.tfidf_recommendations), )
    ok("genre_recommendations",  len(bundle.genre_recommendations))

# 3. 유틸 함수 테스트
def demo_step3_utils():
    section("STEP 3 — 유틸 함수 (UTILS)")

    # _norm_title
    print("\n  [_norm_title]")
    cases = ["  Inception  ", "The DARK Knight", "2001: A Space Odyssey", ""]
    for t in cases:
        ok(f'"{t}"', f'→ "{main._norm_title(t)}"')

    # make_img_url
    print("\n  [make_img_url]")
    paths = ["/abc123.jpg", None, ""]
    for p in paths:
        ok(f'path={p}', main.make_img_url(p))

    # tmdb_card_from_results (async)
    print("\n  [tmdb_card_from_results] (async)")

    async def _run_card():
        fake = [
            {"id": 550,  "title": "Fight Club",  "poster_path": "/a.jpg", "vote_average": 8.4},
            {"id": 27205,"title": "Inception",   "poster_path": "/b.jpg", "vote_average": 8.8},
            {"id": 157336,"title": "Interstellar","poster_path": "/c.jpg","vote_average": 8.6},
        ]
        cards = await main.tmdb_card_from_results(fake, limit=2)
        ok("limit=2, 결과 수", len(cards))
        for c in cards:
            ok(f"  {c.title}", f"id={c.tmdb_id}, poster={c.poster_url}")

    asyncio.run(_run_card())

# 4. pickle 파일 로드 테스트
def demo_step4_pickle():
    section("STEP 4 — Pickle 데이터 로딩 (STARTUP)")

    print("\n  [load_pickles()] 실행 중...")
    try:
        main.load_pickles()
        ok("df 타입",          type(main.df).__name__)
        ok("df 행 수",         len(main.df))
        ok("df 컬럼",          list(main.df.columns))
        ok("tfidf_matrix 크기",main.tfidf_matrix.shape)
        ok("TITLE_TO_IDX 항목 수", len(main.TITLE_TO_IDX))

        print("\n  [df 샘플 — 상위 5개 영화 제목]")
        for i, title in enumerate(main.df["title"].head(5)):
            info(f"  df[{i}]", title)

        print("\n  [TITLE_TO_IDX 샘플 — 상위 5개]")
        for k, v in list(main.TITLE_TO_IDX.items())[:5]:
            info(f'  "{k}"', f"→ idx {v}")

    except FileNotFoundError as e:
        print(f"  [SKIP] pickle 파일 없음: {e}")

#5. TF-IDF 헬퍼 테스트 
def demo_step5_tfidf():
    section("STEP 5 — TF-IDF 헬퍼 함수")

    if main.df is None:
        print("  [SKIP] pickle 미로딩 — STEP 4를 먼저 실행하세요")
        return

    # build_title_to_idx_map
    print("\n  [build_title_to_idx_map]")
    raw_dict = {"Inception": 0, "The Dark Knight": 1, "  Interstellar  ": 2}
    result = main.build_title_to_idx_map(raw_dict)
    ok("dict 입력 → 정규화된 키", dict(list(result.items())[:3]))

    import pandas as pd
    series = pd.Series({"Inception": 0, "Interstellar": 1})
    result2 = main.build_title_to_idx_map(series)
    ok("Series 입력 → 결과", result2)

    # get_local_idx_by_title
    print("\n  [get_local_idx_by_title]")
    first_title = str(main.df.iloc[0]["title"])
    idx = main.get_local_idx_by_title(first_title)
    ok(f'"{first_title}"', f"→ idx {idx}")
    ok(f'"{first_title.upper()}" (대소문자 무관)', main.get_local_idx_by_title(first_title.upper()))

    from fastapi import HTTPException
    try:
        main.get_local_idx_by_title("존재하지않는영화XYZ99999")
    except HTTPException as e:
        ok("없는 제목 → HTTPException", f"status={e.status_code}, detail={e.detail}")

    # tfidf_recommend_titles
    print("\n  [tfidf_recommend_titles]")
    recs = main.tfidf_recommend_titles(first_title, top_n=5)
    ok(f'"{first_title}" top_n=5, 결과 수', len(recs))
    for rank, (title, score) in enumerate(recs, 1):
        info(f"  #{rank}", f"{title}  (score={score:.4f})")

# 전체 실행 함수
if __name__ == "__main__":
  print("=" * 60)

  # demo_step1_env()
  # demo_step2_models()
  # demo_step3_utils()
  demo_step4_pickle()
  demo_step5_tfidf()

  print("=" * 60)

