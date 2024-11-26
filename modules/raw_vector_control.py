from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
import os
import dotenv
from datetime import datetime  # 날짜 변환에 사용

# 환경 변수 로드
dotenv.load_dotenv()

# Qdrant 클라이언트 초기화
qdrant_client = QdrantClient(
    url="https://b236ccb3-b4d5-4349-a48d-6a8026f62951.us-east4-0.gcp.cloud.qdrant.io",
    api_key=os.environ["QDRANT_API_KEY"]
)


def format_date(unix_time):
    return f"(Date: {datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d')})"


def get_raw_memory(user_id="testman1", start_date: int = None, end_date: int = None, limit: int = 30):
    # 컬렉션 이름 설정
    collection_name = "CSD1"

    # 특정 user_id로 필터링 조건 생성
    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",  # 필터링할 필드 이름
                match=MatchValue(value=user_id)  # 매칭 조건
            ),
            FieldCondition(
                key="created_at",  # 필터링할 필드 이름
                range=Range(
                    gt=start_date,  # 초과 조건 (없음)
                    lt=end_date,  # 미만 조건 (없음)
                )
            )
        ]
    )

    points, _ = qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=scroll_filter
    )

    # 필요한 필드만 추출하여 저장
    filtered_points = []
    for point in points:
        filtered_points.append({
                "data": point.payload.get("data"),
                "created_at": format_date(point.payload.get('created_at'))
            })

    return filtered_points

get_raw_memory()