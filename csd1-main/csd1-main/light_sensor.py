import smbus
import asyncio

# BH1750 주소 및 명령어
BH1750_ADDRESS = 0x23
BH1750_ONE_TIME_HIGH_RES_MODE = 0x20

# I2C 버스 초기화
bus = smbus.SMBus(1)

async def read_light_async(feeling_threshold=100):
    """
    조도 값을 읽어 밝기 상태를 반환
    :param feeling_threshold: 조도 감정 변화 기준 값
    :return: (lux, 상태 값: 1 밝음, 0 어두움)
    """
    data = await asyncio.to_thread(bus.read_i2c_block_data, BH1750_ADDRESS, BH1750_ONE_TIME_HIGH_RES_MODE)
    light_level = (data[0] << 8) | data[1]
    lux = light_level / 1.2
    print(f"조도 값: {lux:.2f} Lux")
    return lux, 1 if lux > feeling_threshold else 0
