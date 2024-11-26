#import smbus
import asyncio
#import serial
import struct

from sensor_class import Sensor

# I2C 버스 초기화
#bus = smbus.SMBus(1)

# UART 초기화
k = struct.pack('B', 0xff)
#ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)

# BH1750 주소 및 명령어
BH1750_ADDRESS = 0x23
BH1750_ONE_TIME_HIGH_RES_MODE = 0x20


# UART 초기화
k = struct.pack('B', 0xff)

# 비동기로 이미지를 변경하는 함수 -> 아마 나중에 이걸 따로 빼서 이미지변경 코드 만들어야 할 것 -> 그러면 얘를 아예 센서 클래스로 옮기는게 좋음
async def change_image_async(image_id):
    """
    # 주석 처리된 실제 하드웨어 코드 보존
    if image_id == 0:
        await asyncio.to_thread(ser.write, b"pic 40,0,0")
    elif image_id == 1:
        await asyncio.to_thread(ser.write, b"pic 40,0,1")
    elif image_id == 2:
        await asyncio.to_thread(ser.write, b"pic 40,0,2")
    elif image_id == 3:
        await asyncio.to_thread(ser.write, b"pic 40,0,3")
    else:
        print("유효하지 않은 이미지 ID입니다.")
        return
    # UART 종료 명령 전송
    await asyncio.to_thread(ser.write, k * 3)
    """

# 비동기로 조도 값을 읽는 함수
async def read_light_async(sensor : Sensor):
    """
    # 주석 처리된 실제 하드웨어 코드 보존
    data = await asyncio.to_thread(bus.read_i2c_block_data, BH1750_ADDRESS, BH1750_ONE_TIME_HIGH_RES_MODE)
    light_level = (data[0] << 8) | data[1]
    lux = light_level / 1.2
    print(f"조도 값: {lux:.2f} Lux")
    """
    # 시뮬레이션용 조도 값 생성
    lux = 150  # 임의의 조도 값
    feeling_threshold = 100  # 감정 변경 임계값

    # 조도 값에 따라 감정 설정
    if lux > feeling_threshold:
        sensor.set_feeling(1)  # 밝으면 감정 1
        await change_image_async(0)
    else:
        sensor.set_feeling(0)  # 어두우면 감정 0
        await change_image_async(1)
