import asyncio
import serial
import struct

# UART 초기화
ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)
k = struct.pack('B', 0xff)

async def change_image_async(image_id):
    """
    이미지 변경을 수행
    :param image_id: 변경할 이미지 ID
    """


    commands = {
        0: b"pic 40,0,0",
        1: b"pic 40,0,1",
        2: b"pic 40,0,2",
        3: b"pic 40,0,3"
    }
    if image_id in commands:
        await asyncio.to_thread(ser.write, commands[image_id])
        await asyncio.to_thread(ser.write, k * 3)
    else:
        print("유효하지 않은 이미지 ID입니다.")

    return 0
