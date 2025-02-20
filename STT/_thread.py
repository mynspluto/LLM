import threading
import time


# 쓰레드에서 실행할 함수
def worker(thread_num):
    print(f"쓰레드 {thread_num} 시작")
    # 실제 작업을 시뮬레이션하기 위한 대기
    time.sleep(2)
    print(f"쓰레드 {thread_num} 종료")


# 메인 코드
def main():
    # 여러 개의 쓰레드 생성
    threads = []
    for i in range(3):
        # 쓰레드 생성
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        # 쓰레드 시작
        t.start()

    # 모든 쓰레드가 종료될 때까지 대기
    for t in threads:
        t.join()

    print("모든 쓰레드 작업 완료")


# 프로그램 실행
if __name__ == "__main__":
    main()
