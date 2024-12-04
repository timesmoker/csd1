#from gpiozero import DigitalInputDevice
import asyncio

# 토양센서 GPIO 핀 번호 설정
DO_PIN = 25

async def read_moisture():
    """
    토양 수분 상태를 반환
    :return: 토양 상태 값: 1 건조, 0 촉촉
    """

    """
    soil_sensor = DigitalInputDevice(DO_PIN)
    state = soil_sensor.value  # 1: 건조, 0: 촉촉
    print(f"토양 상태: {'건조' if state == 1 else '촉촉'}")
    return state
    """
    return 1