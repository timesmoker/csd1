# 센서 클래스 -> 센서로 받은 정보들에 대해 각 쓰레드가 공유할 수 있도록 존재,
# 감정 정보를 저장하고, 감정 정보를 가져오는 단순한 기능만 잇음
# 향후 필요하다면 해당 센서 클래스에 있는 감정 기준으로 화분 표정이 바뀌도록 할 수 있음

class Sensor:
    def __init__(self, feeling=1):
        self.feeling = feeling  # 초기 감정 상태

    def set_feeling(self, feeling):
        self.feeling = feeling

    def get_feeling(self):
        return self.feeling

#   def calculate_feeling(self):
#   나중에 여기 채워서 감정 계산 시키고, return 한 다음에 감정에 따라 화분 표정 바뀌게 하면 좋을것