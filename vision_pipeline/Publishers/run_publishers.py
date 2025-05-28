# run_publishers.py
from multiprocessing import Process
from Publishers import LArmPublisher, HeadPublisher

if __name__ == "__main__":
    p1 = Process(target=LArmPublisher.run)
    p2 = Process(target=HeadPublisher.run)

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()