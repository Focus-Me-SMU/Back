import time

class EyeTrackingCycle:
    def __init__(self, sentence_count):
        self.sentence_count = sentence_count  # 페이지의 문장 수
        self.current_state = 0  # 시선의 현재 상태
        self.start_time = None  # 한 문장 읽기 시작 시간
        self.program_start_time = time.time()  # 프로그램 가동 시작 시간
        self.total_cycle_time = 0  # 모든 문장을 읽는데 걸린 총 시간
        self.cycle_start_time = None  # 각 사이클의 시작 시간
        self.cycle_times = []  # 각 사이클의 시간을 저장
        self.sentence_times = []  # 각 문장을 읽는 시간을 저장
        self.in_progress = False  # 사이클 진행 여부
        self.cycles_completed = 0  # 완료된 사이클 수
        self.time_logs = []  # 각 사이클의 시작 및 종료 시간 기록

    def update_state(self, eye_region):
        if eye_region == 0 and not self.in_progress:
            # 새로운 사이클 시작
            self.cycle_start_time = time.time()
            self.in_progress = True
            self.current_state = 0
            self.time_logs.append((self.cycle_start_time, "Start cycle"))
        elif eye_region == self.current_state + 1:
            # 상태가 순차적으로 변경되는 경우 (0 -> 1, 1 -> 2, 2 -> 3)
            self.current_state = eye_region
            if self.current_state == 3:
                # 사이클 완료
                cycle_end_time = time.time()
                cycle_duration = cycle_end_time - self.cycle_start_time
                self.cycle_times.append(cycle_duration)
                self.total_cycle_time += cycle_duration
                self.cycles_completed += 1
                self.time_logs.append((cycle_end_time, f"End cycle {self.cycles_completed}"))

                # 사이클이 끝나면 문장 읽기 시간 기록
                self.sentence_times.append(cycle_duration)

                # 사이클 완료 후 초기화
                self.in_progress = False
                self.current_state = 0

        elif eye_region == 0:
            # 새로운 문장 시작
            self.current_state = 0

    def get_program_running_time(self):
        return time.time() - self.program_start_time

    def get_total_cycle_time(self):
        return self.total_cycle_time

    def get_average_time_per_sentence(self):
        if len(self.sentence_times) > 0:
            return sum(self.sentence_times) / len(self.sentence_times)
        return 0

    def get_time_logs(self):
        return self.time_logs

    def all_cycles_completed(self):
        return self.cycles_completed >= self.sentence_count

    def get_detailed_logs(self):
        program_end_time = time.time()
        logs = [f"Program started at: {time.strftime('%H:%M:%S', time.localtime(self.program_start_time))}"]
        for log_time, description in self.time_logs:
            logs.append(f"{description} at {time.strftime('%H:%M:%S', time.localtime(log_time))}")
        logs.append(f"Program ended at: {time.strftime('%H:%M:%S', time.localtime(program_end_time))}")
        return logs