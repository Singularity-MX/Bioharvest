from apscheduler.schedulers.background import BackgroundScheduler
from src.core.experiment_runner import run_capture_cycle

scheduler = BackgroundScheduler()


def start_scheduler():

    for hora in range(24):
        scheduler.add_job(
            run_capture_cycle,
            "cron",
            hour=hora,
            minute=0,
            args=[hora]
        )

    scheduler.start()
    print("Scheduler iniciado")


def stop_scheduler():
    scheduler.shutdown()
