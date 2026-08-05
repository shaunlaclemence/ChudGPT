from apscheduler.schedulers.asyncio import AsyncIOScheduler


def flush_and_reset_quotas():
    pass


scheduler = AsyncIOScheduler()
scheduler.add_job(flush_and_reset_quotas, "interval", hours=1)
scheduler.start()
