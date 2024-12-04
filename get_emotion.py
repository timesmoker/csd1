import asyncio
from light_sensor import read_light_async
from soil_sensor import read_moisture
from image_manager import change_image_async

async def get_emotion():
    """
    센서 데이터를 기반으로 감정을 결정하고 이미지 변경
    """
    lux, light_state = await read_light_async()  # 조도 값과 상태
    soil_state = await read_moisture()  # 토양 상태

    if light_state == 1 and soil_state == 1:
        # 조도 충분, 토양 건조
        await change_image_async(0)
    elif light_state == 0 and soil_state == 0:
        # 조도 부족, 토양 촉촉
        await change_image_async(1)
    elif light_state == 1 and soil_state == 0:
        # 조도 충분, 토양 촉촉
        await change_image_async(2)
    elif light_state == 0 and soil_state == 1:
        # 조도 부족, 토양 건조
        await change_image_async(3)

if __name__ == "__main__":
    asyncio.run(get_emotion())