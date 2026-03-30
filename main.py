import os
import pickle
import numpy as np
import pandas as pd
import httpx # 비동기 http 클라이언트: tmdb api 호출에 사용

from contextlib import asynccontextmanager # FastAPI의 수명 주기를 관리하기 위한 비동기 컨텍스트 관리자
from typing import Optional, List, Dict, Any, Tuple # 타입 힌트를 위한 모듈
from fastapi import FastAPI, HTTPException, Query # FastAPI 프레임워크의 핵심 모듈
from fastapi.middleware.cors import CORSMiddleware # CORS(Cross-Origin Resource Sharing)를 처리하기 위한 미들웨어
from pydantic import BaseModel # 데이터 유효성 검사를 위한 Pydantic 모델
from dotenv import load_dotenv # .env 파일에서 환경 변수를 로드하기 위한 모듈

# env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = 'https://api.themoviedb.org/3'
TMDB_IMG_500 = 'https://image.tmdb.org/t/p/w500'

if not TMDB_API_KEY:
    raise RuntimeError("TMDB API 키가 없습니다.")

# Fast Api 기본 설정
@asynccontextmanager
async def lifespan(app: FastAPI):
  load_pickles()
  yield

app = FastAPI(title="Movie Recommender API", version="1.0", lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

# pickle 파일 가져오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 절대 경로 설정(root)

DF_PATH = os.path.join(BASE_DIR, "data", "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "data", "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "data", "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "data", "tfidf.pkl")

# 변수 및 타입 초기화
df: Optional[pd.DataFrame] = None
indices_obj: Any = None 

tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None 

# 리턴되는 데이터 구조 타입 정의
class TMDBMovieCard(BaseModel):
  tmdb_id: int
  title: str
  poster_url: Optional[str] = None
  release_date: Optional[str] = None
  vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
  tmdb_id: int
  title: str
  overview: Optional[str] = None
  release_date: Optional[str] = None
  poster_url: Optional[str] = None
  backdrop_url: Optional[str] = None
  genres: Optional[List[dict]] = None

class TFIDFRecItem(BaseModel):
  title: str
  score: float
  tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
  query: str
  movie_details: TMDBMovieDetails
  tfidf_recommendations: List[TFIDFRecItem]
  genre_recommendations: List[TMDBMovieCard]

# 사용 함수 정의
def _norm_title(t: str) -> str:
  # 영화 제목을 정규화: 앞뒤 공백 제거 후 소문자로 변환하여 비교에 사용
  return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
  # TMDB 이미지 경로를 완전한 URL로 변환. 경로가 없으면 None 반환
  if not path:
    return None
  return f"{TMDB_IMG_500}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
  """
  TMDB API에 안전하게 GET 요청을 보내는 함수:
  - 네트워크 오류 발생 시 -> 502 에러 반환
  - TMDB API 오류 발생 시 -> 502 에러와 상세 메시지 반환
  """
  q = dict(params)
  q["api_key"] = TMDB_API_KEY

  try:
    async with httpx.AsyncClient(timeout=20) as client: # timeout=20 seconds
      r = await client.get(f"{TMDB_BASE}{path}", params=q)
  except httpx.RequestError as e:
    raise HTTPException(
      status_code=502,
      detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
    )
  
  if r.status_code !=200:
    raise HTTPException(
      status_code=502,
      detail=f"TMDB error {r.status_code} | {r.text}"
    )

  return r.json()

async def tmdb_card_from_results(results: List[dict], limit: int=20) -> List[TMDBMovieCard]:
  # TMDB 검색 결과 목록을 TMDBMovieCard 객체 리스트로 변환. limit 개수만큼만 처리
  out: List[TMDBMovieCard] = []
  for m in (results or [])[:limit]:
    out.append(
      TMDBMovieCard(
        tmdb_id=int(m["id"]),
        title=m.get("title") or m.get("name") or "",
        poster_url=make_img_url(m.get("poster_path")),
        release_date=m.get("release_date"),
        vote_average=m.get("vote_average")
      )
    )

  return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
  # TMDB 영화 ID로 상세 정보(개요, 장르, 포스터 등)를 조회하여 TMDBMovieDetails 객체로 반환
  data = await tmdb_get(f"/movie/{movie_id}", {"language": "ko-KR"})
  return TMDBMovieDetails(
    tmdb_id=int(data["id"]),
    title=data.get("title") or "",
    overview=data.get("overview"),
    release_date=data.get("release_date"),
    poster_url=make_img_url(data.get("poster_path")),
    backdrop_url=make_img_url(data.get("backdrop_path")),
    genres=data.get("genres", []) or []
  )

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
  """
  키워드로 TMDB 영화를 검색하여 원본 응답(여러 결과 포함)을 반환하는 함수.
  """
  return await tmdb_get(
    "/search/movie",
    {
      "query": query,
      "include_adult": "false",
      "language": "ko-KR",
      "page": page,
    }
  )

async def tmdb_search_first(query: str) -> Optional[dict]:
  # 키워드로 TMDB를 검색하여 첫 번째 결과만 반환. 결과가 없으면 None 반환
  data = await tmdb_search_movies(query=query, page=1)
  results = data.get("results", [])
  return results[0] if results else None

# TF-IDF 관련 함수
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
  """
  indices.pkl 데이터를 정규화된 제목->인덱스 딕셔너리로 변환하는 함수.
  지원 형식:
  - dict(제목 -> 인덱스)
  - pandas Series (인덱스=제목, 값=인덱스)
  결과를 TITLE_TO_IDX에 사용할 수 있도록 정규화하여 반환.
  """
  title_to_idx: Dict[str, int] = {}

  if isinstance(indices, dict):
    for k, v in indices.items():
      title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx

  # pandas Series or similar mapping
  try:
    for k, v in indices.items():
      title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx
  except Exception:
    # last resort: if it's a list-like etc
    raise RuntimeError(
      "indices.pkl must be dict or pandas Series-like (with .items())"
    )

# TF-IDF 행렬 (tfidf_matrix)
#   행 0  → "The Dark Knight"
#   행 1  → "Inception"
#   행 2  → "Interstellar"
#   ...

# TITLE_TO_IDX (딕셔너리)
#   "the dark knight" → 0
#   "inception"       → 1
#   "interstellar"    → 2
# TITLE_TO_IDX는 build_title_to_idx_map()이 indices.pkl 파일을 읽어 만든 전역 딕셔너리입니다.

# 사용자 요청: "Inception"
#         ↓
# get_local_idx_by_title("Inception")
#         ↓
# _norm_title → "inception"
#         ↓
# TITLE_TO_IDX["inception"] → 1
#         ↓
# tfidf_matrix[1] 행을 기준으로 유사 영화 계산

def get_local_idx_by_title(title: str) -> int:
  # 영화 제목(문자열) 을 받아서, TF-IDF 행렬에서 그 영화가 몇 번째 행(row)인지 정수 인덱스 를 반환
  global TITLE_TO_IDX 
  if TITLE_TO_IDX is None:
    raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")

  # 예: "  Inception  " → "inception"
  key = _norm_title(title)
  if key in TITLE_TO_IDX:
    return int(TITLE_TO_IDX[key])
  raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")

def tfidf_recommend_titles(query_title: str, top_n=10) -> List[Tuple[str, float]]:
  """
  TF-IDF matrix에 따른 코사인 유사도를 사용하여 local df로부터 (타이틀, 스코어) 리스트 반환
  """

  global df, tfidf_matrix
  if df is None or tfidf_matrix is None:
    raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

  idx = get_local_idx_by_title(query_title)

  # query vector
  qv = tfidf_matrix[idx]
  # 희소 행렬 -> 일반 행렬(toarray()) -> 1차원 행렬(revel())
  scores = (tfidf_matrix @ qv.T).toarray().ravel()

  # sort descending
  order = np.argsort(-scores)

  out: List[Tuple[str, float]] = []
  for i in order:
    if int(i) == int(idx):
      continue
    try:
      title_i = str(df.iloc[int(i)]["title"])
    except Exception:
      continue
    out.append((title_i, float(scores[int(i)])))
    if len(out) >= top_n:
      break
  return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
  """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
  """

  try:
    m = await tmdb_search_first(title)
    if not m:
      return None
    return TMDBMovieCard(
      tmdb_id=int(m["id"]),
      title=m.get("title") or title,
      poster_url=make_img_url(m.get("poster_path")),
      release_date=m.get("release_date"),
      vote_average=m.get("vote_average")
    )

  except Exception:
    return None

# Load Pickle Files
def load_pickles():
  global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

  # load df
  with open(DF_PATH, "rb") as f:
    df = pickle.load(f)

  # load indices
  with open(INDICES_PATH, "rb") as f:
    indices_obj = pickle.load(f)

  # load TF-IDF matrix
  with open(TFIDF_MATRIX_PATH, "rb") as f:
    tfidf_matrix = pickle.load(f)

  # load tfidf vectorizer (optional, not used directly here)
  with open(TFIDF_PATH, "rb") as f:
    tfidf_obj = pickle.load(f)

  # build normalized map
  TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

  # sanity
  if df is None or "title" not in df.columns:
    raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

# FastAPI endpoint

# 초기 화면 로드 시 엔드포인트
@app.get('/home', response_model=List[TMDBMovieCard])
async def home(
  category: str = Query('trending'),
  limit: int = Query(20, ge=1, le=50) # 기본값, 최소값, 최대값
):
  """
  인기순으로 영화 목록 추출
  """
  try:
    if category == "trending":
      data = await tmdb_get("/trending/movie/day", {"language": "ko-KR"})
      return await tmdb_card_from_results(data.get('results', []), limit=limit)

    if category not in {'popular', 'top_rated', 'upcoming', 'now-playing'}:
      raise HTTPException(status_code=400, detail="Invalide category")

    data = await tmdb_get(f'/movie/{category}, {"language": "ko-KR", "page": 1}')
    return await tmdb_card_from_results(data.get('results', []), limit=limit)
  except HTTPException:
    raise HTTPException(status_code=502, detail="TMDB request error")

# 키워드 검색
@app.get('/tmdb/search')
async def tmdb_search(query: str=Query(..., min_length=1), page: int=Query(1, ge=1, le=10)):
  """
  키워드로 영화 검색
  """
  return await tmdb_search_movies(query=query, page=page)

# movie details 엔드포인트
@app.get('/movie/id/{tmdb_id}', response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
  """
  TMDB 아이디로 영화 상세 정보 조회
  """
  return await tmdb_movie_details(tmdb_id)

# 장르 추천 엔드포인트: TODO - 같은 영화만 나오는 부분 수정
@app.get('/recommend/genre', response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: int=Query(...), limit: int=Query(10, ge=1, le=20)):
  """
  TMDB 아이디를 기반으로 장르 추천
  - details 정보 조회
  - 첫 번째 장르 선택
  - 주어진 장르로 부터 인기순으로 영화 목록 반환
  """
  details = await tmdb_movie_details(tmdb_id)
  if not details.genres:
    return []

  genre_id = details.genres[0]['id']
  discover = await tmdb_get('/discover/movie', {
    "with_genre": genre_id,
    "language": "ko-KR",
    "sort_by": 'popularity.desc',
    "page": 1
  })

  cards = await tmdb_card_from_results(discover.get('results', []), limit=limit)
  return [c for c in cards if c.tmdb_id != tmdb_id] # 자신과 같은 영화 제외

# TF_IDF 확인용 엔드포인트
@app.get('/recommend/tfidf')
async def tfidf_recommend(title: str=Query(..., min_length=1), top_n: int=Query(10, ge=1, le=10)):
  """
  TF-IDF 기반 영화 추천 스코어 확인
  """
  recs = tfidf_recommend_titles(title, top_n=top_n)
  return [{"title": t, "score": s} for t, s in recs]

# 번들용 추천 종합: details + tfidf + genre recommend
@app.get('movie/search', response_model=SearchBundleResponse)
async def search_bundle(
  query: str = Query(..., min_length=1),
  tfidf_top_n: int = Query(10, ge=1, le=20),
  genre_top_n: int = Query(10, ge=1, le=20),
):
  """
  입력한 영화에 대해
  - 영화 상세 정보
  - TF-IDF 기반 추천(로컬) + 포스터
  - 장르 기반 추천(TMDB) + 포스터
  """
  pass

if __name__ == "__main__":
  import uvicorn
  uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)
