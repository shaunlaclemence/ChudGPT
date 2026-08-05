import functools

from sqlalchemy.exc import IntegrityError


def db_exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except IntegrityError as err:
            raise ValueError("DB Error: ", err)

        return result

    return wrapper
